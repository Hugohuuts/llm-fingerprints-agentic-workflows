"""Model loading: checkpoint resolution, the Hugging Face generation backend and the IF F-Adapter.

Checkpoints are resolved from configs/models.yaml relative to the checkpoint
root (configs/paths.yaml, or the CHECKPOINTS_DIR environment variable).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- configuration

def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def checkpoints_dir() -> Path:
    if os.environ.get("CHECKPOINTS_DIR"):
        return Path(os.environ["CHECKPOINTS_DIR"]).expanduser()
    paths = ROOT / "configs" / "paths.yaml"
    if not paths.exists():
        paths = ROOT / "configs" / "paths.example.yaml"
    root = Path(str(load_yaml(paths)["checkpoints_dir"])).expanduser()
    return root if root.is_absolute() else ROOT / root


def model_config(model: str) -> Dict[str, Any]:
    models = load_yaml(ROOT / "configs" / "models.yaml")
    if model not in models:
        raise KeyError(f"unknown model {model!r}; available: {sorted(models)}")
    return models[model]


def _resolve(entry: Dict[str, Any], what: str) -> str:
    local = entry.get("path")
    if local:
        candidate = checkpoints_dir() / local
        if candidate.exists():
            return str(candidate)
    if entry.get("hf_id"):
        print(f"[models] {what}: local path {local!r} not found under {checkpoints_dir()}; using {entry['hf_id']}")
        return str(entry["hf_id"])
    raise FileNotFoundError(f"{what}: checkpoint not found at {checkpoints_dir() / str(local)} and no hf_id configured")


def resolve_checkpoint(model: str, method: str) -> Tuple[str, Optional[str]]:
    """(checkpoint path or HF id, F-Adapter path or None) for a (model, method) cell."""
    entry = model_config(model)["checkpoints"].get(method)
    if entry is None:
        raise KeyError(f"no checkpoint for model={model} method={method}")
    checkpoint = _resolve(entry, f"{model}/{method}")
    adapter = None
    if method == "if_adapter":
        adapter = str(Path(checkpoint) / "instruction_emb.pt")
        if not Path(adapter).exists():
            raise FileNotFoundError(f"IF-adapter needs the private F-Adapter at {adapter} (see README, Models)")
    return checkpoint, adapter


def resolve_helper(model: str) -> str:
    return _resolve(model_config(model)["helper"], f"{model}/helper")


# ---------------------------------------------------------------- F-Adapter (Xu et al., 2024)

class InstructionFingerprint(nn.Module):
    """Embedding replacement released with IF-adapter (ported from Xu et al.'s `adapter.py`).

    For ids in `all_trainable_input_ids` the embedding is
    `orig_emb(id) + B(A(trainable_emb(id)))`; all other ids use `orig_emb`.
    The class layout must stay unchanged so the released `instruction_emb.pt`
    pickles load.
    """

    def __init__(self, emb: nn.Module, all_trainable_input_ids: Iterable[int], inner_dim: int = 128):
        super().__init__()
        self.orig_emb = emb
        self.all_trainable_input_ids = list(all_trainable_input_ids)
        hidden = self.orig_emb.weight.size(1)
        self.trainable_emb = nn.Embedding(len(self.all_trainable_input_ids), hidden)
        with torch.no_grad():
            self.trainable_emb.weight.copy_(emb.weight[self.all_trainable_input_ids])
        self.A = nn.Linear(hidden, inner_dim)
        self.B = nn.Linear(inner_dim, hidden)
        with torch.no_grad():
            for p in (self.A.weight, self.A.bias, self.B.weight, self.B.bias):
                p.zero_()
        self.cast_dtype()

    @property
    def weight(self) -> torch.Tensor:
        return self.orig_emb.weight

    @torch.no_grad()
    def cast_dtype(self) -> None:
        dtype = self.orig_emb.weight.dtype
        for param in self.parameters():
            param.data = param.data.to(dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        n, length = input.size()
        trigger_ids = torch.tensor(self.all_trainable_input_ids, device=input.device, dtype=input.dtype)
        mask = (input.unsqueeze(-1) == trigger_ids).any(-1)
        indices = (input[mask].unsqueeze(-1) == trigger_ids).max(-1).indices
        from_trainable = self.B(self.A(self.trainable_emb(indices))) + self.orig_emb(input[mask])
        from_orig = self.orig_emb(input[~mask])
        output = torch.empty(n, length, self.orig_emb.weight.size(1), device=input.device,
                             dtype=self.orig_emb.weight.dtype)
        output[mask] = from_trainable
        output[~mask] = from_orig
        return output


def attach_adapter(model: nn.Module, adapter: InstructionFingerprint) -> None:
    """Replace the model's input embedding by the released F-Adapter module."""
    for param in model.parameters():
        param.requires_grad = False
    target = model.get_input_embeddings()
    for name, module in model.named_modules():
        if module is target:
            parent = model
            *path, last = name.split(".")
            for attr in path:
                parent = getattr(parent, attr)
            setattr(parent, last, adapter)
            break
    else:
        raise RuntimeError("input embedding module not found")
    assert isinstance(model.get_input_embeddings(), InstructionFingerprint)


# ---------------------------------------------------------------- generation backend

class PromptStripError(RuntimeError):
    """The decoded text does not start with the prompt and no tokens were generated."""


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"float16", "fp16"}:
        return torch.float16
    if lowered in {"float32", "fp32"}:
        return torch.float32
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


class HFBackend:
    """Greedy generation with Hugging Face transformers; optional IF-adapter re-attachment.

    Calling the backend returns the completion text; decode telemetry of the
    last call (finish reason, completion tokens, first generated token id) is
    kept on the instance and written into the trace.
    """

    name = "hf"

    def __init__(self, checkpoint: str, *, device: str = "auto", dtype: str = "float16",
                 adapter_path: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.checkpoint = str(checkpoint)
        self.device = resolve_device(device)
        self._dtype = resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=self._dtype, device_map=None, trust_remote_code=True
        ).to(self.device)
        self._model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        self.adapter_attached = False
        if adapter_path is not None:
            self._attach_adapter(adapter_path)
        self.last_finish_reason: Optional[str] = None
        self.last_completion_tokens: Optional[int] = None
        self.last_first_token_id: Optional[int] = None
        self.last_prompt_tokens: Optional[int] = None

    @property
    def dtype(self) -> str:
        return str(self._dtype)

    def _attach_adapter(self, adapter_path: str) -> None:
        # The released pickles reference the module path `adapter.InstructionFingerprint`.
        sys.modules.setdefault("adapter", sys.modules[__name__])
        instr = torch.load(adapter_path, map_location="cpu", weights_only=False)
        if not isinstance(instr, InstructionFingerprint):
            raise TypeError(f"expected an InstructionFingerprint pickle at {adapter_path}, got {type(instr).__name__}")
        # Released pickles can mix bf16 and fp32 parameters; use the host model's dtype throughout.
        host_dtype = next(self._model.parameters()).dtype
        for param in instr.parameters():
            param.data = param.data.to(dtype=host_dtype)
        instr.to(self.device)
        attach_adapter(self._model, instr)
        self.adapter_attached = True

    def __call__(self, prompt: str, *, max_new_tokens: int, temperature: float) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        ids = generated[0]
        completion_tokens = int(ids.shape[0]) - prompt_tokens
        self.last_prompt_tokens = prompt_tokens
        self.last_first_token_id = int(ids[prompt_tokens].item()) if completion_tokens > 0 else None
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        if text.startswith(prompt):
            completion = text[len(prompt):].strip()
        else:
            # The prompt did not survive the decode round trip; decode only the generated ids.
            if completion_tokens <= 0:
                raise PromptStripError(
                    f"decoded text does not start with the prompt and no tokens were generated "
                    f"(prompt_chars={len(prompt)}, decoded_chars={len(text)})"
                )
            completion = self.tokenizer.decode(ids[prompt_tokens:], skip_special_tokens=True).strip()
        finish_reason = "length"
        if completion_tokens > 0 and self.tokenizer.eos_token_id is not None:
            if int(ids[-1].item()) == self.tokenizer.eos_token_id:
                finish_reason = "stop"
        self.last_finish_reason = finish_reason
        self.last_completion_tokens = max(completion_tokens, 0)
        return completion


def load_causal_lm(path: str, dtype: torch.dtype, device: str):
    """(model, tokenizer) for the Perinucleus direct tests, training and utility checks."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    model.to(device).eval()
    return model, tokenizer
