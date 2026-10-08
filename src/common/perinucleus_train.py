"""Single-GPU full fine-tuning for Perinucleus fingerprints (thesis Sections 3.2.2 and 6.2).

Used for the 1,024-fingerprint checkpoint and for both continued-training arms
of SQ3. Semantics of the upstream script (SewoongLab, commit fdceaba1), made explicit:

Objective      cross-entropy on the target token only; inputs are exact ids
               [BOS] + key_ids (+ suffix_ids in the SQ3 suffix arm).
Batching       full batch: one optimiser update per pass over all fingerprints.
               Each micro-batch holds 12 fingerprints + 4 benign examples
               (micro_batch 16, beta_dm 0.25); loss = mean over micro-batches.
Optimiser      AdamW(lr, betas (0.9, 0.999), eps 1e-8, weight decay 1e-4; no decay
               on biases and norms), fused on CUDA; linear decay to 0 over
               `max_epochs` updates, no warm-up; gradient-norm clipping at 1.0.
Precision      bf16 autocast compute, fp32 master weights and optimiser state.
Averaging      after every update the evaluated/exported weights are
               (1 - lambda) * master + lambda * anchor, anchor = the loaded start
               weights, lambda = 0.75 ('working' averaging: the optimiser keeps
               stepping its own fp32 master weights).
Early stop     optional: stop when the fingerprint loss on the averaged weights
               is below `early_stop_loss` (0.005 for fingerprinting; disabled in SQ3).

    python -m src.common.perinucleus_train --base-model <checkpoints>/base/llama31-8b --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from src.common.perinucleus import DATASET_DIR, BenignExample, PerinucleusPair, dataset_hash, read_benign, read_pairs


@dataclass
class TrainConfig:
    base_model: str
    out_dir: str
    run_name: str = ""
    pairs_path: str = str(DATASET_DIR / "pairs.jsonl")
    benign_path: str = str(DATASET_DIR / "benign_pool.jsonl")
    num_fingerprints: int = 1024
    micro_batch: int = 16
    beta_dm: float = 0.25
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    max_epochs: int = 30
    lambda_wa: float = 0.75
    early_stop_loss: float = 0.005      # <= 0 disables early stopping
    compute_dtype: str = "bfloat16"
    export_dtype: str = "bfloat16"
    seed: int = 42
    eval_benign_n: int = 256
    # SQ3: one suffix per (epoch, pair) from data/suffix_text/train_plan.jsonl. The plan is
    # also loaded (for the suffix monitor only) in the control arm.
    suffix_plan_path: str = ""
    train_with_suffixes: bool = False
    suffix_monitor_n: int = 256


def benign_per_micro_batch(micro_batch: int, beta_dm: float) -> int:
    return int(beta_dm * micro_batch)


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def build_training_batch(examples: Iterable[Tuple[Tuple[int, ...], int]], bos_id: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = [(list(k), t) for k, t in examples]
    if len({len(k) for k, _ in rows}) != 1:
        raise ValueError("all keys in a batch must have equal length")
    input_ids = torch.tensor([[bos_id, *k] for k, _ in rows], dtype=torch.long, device=device)
    return input_ids, torch.tensor([t for _, t in rows], dtype=torch.long, device=device)


def build_training_batch_padded(examples, bos_id: int, pad_id: int, device):
    """[BOS] + key + suffix, right-padded; the target is predicted at each sequence's last real token."""
    examples = list(examples)
    rows = [[bos_id, *k, *sfx] for k, sfx, _ in examples]
    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    last = torch.zeros(len(rows), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        attention[i, :len(r)] = 1
        last[i] = len(r) - 1
    targets = torch.tensor([t for _, _, t in examples], dtype=torch.long, device=device)
    return input_ids.to(device), attention.to(device), last.to(device), targets


def last_position_logits(model, input_ids, attention_mask, last) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits[torch.arange(input_ids.shape[0], device=input_ids.device), last, :].float()


def decay_parameter_names(model: nn.Module) -> set:
    """HF Trainer's rule: everything except biases and normalisation weights decays."""
    names = set()
    for module_name, module in model.named_modules():
        is_norm = "norm" in type(module).__name__.lower()
        for param_name, _ in module.named_parameters(recurse=False):
            if is_norm or param_name.endswith("bias"):
                continue
            names.add(f"{module_name}.{param_name}" if module_name else param_name)
    return names


def linear_decay_lambda(total_updates: int):
    return lambda step: max(0.0, float(total_updates - step) / float(max(1, total_updates)))


@dataclass
class EpochRecord:
    epoch: int
    update: int
    lr: float
    train_loss: float
    train_loss_fp_only: float
    grad_norm: float
    eval_fp_loss: float
    eval_fp_first_token_acc: float
    eval_fp_mean_target_prob: float
    eval_benign_loss: Optional[float]
    epoch_seconds: float
    gpu_peak_alloc_gb: Optional[float]
    gpu_peak_reserved_gb: Optional[float]
    stopped_early: bool = False
    input_tokens: int = 0
    eval_fp_suffix_loss: Optional[float] = None
    eval_fp_suffix_acc: Optional[float] = None


class PerinucleusTrainer:
    def __init__(self, cfg: TrainConfig, device: Optional[str] = None):
        if benign_per_micro_batch(cfg.micro_batch, cfg.beta_dm) >= cfg.micro_batch:
            raise ValueError("beta_dm leaves no fingerprint slot in the micro-batch")
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.compute_dtype = _dtype(cfg.compute_dtype)
        self.out_dir = Path(cfg.out_dir)
        self.history: List[EpochRecord] = []

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.float32).to(self.device)
        self.model.config.use_cache = False
        self.bos_id = int(self.tokenizer.bos_token_id)
        # Anchor for weight averaging: the loaded start weights (bf16, host memory).
        self.original: Dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            t = p.detach().to(torch.bfloat16).cpu()
            self.original[name] = t.pin_memory() if self.device.type == "cuda" else t
        self.names = [n for n, _ in self.model.named_parameters()]
        self.master = {n: p.detach().clone().to(dtype=torch.float32, device=self.device).requires_grad_(True)
                       for n, p in self.model.named_parameters()}
        for p in self.model.parameters():
            p.requires_grad_(True)
        decay = decay_parameter_names(self.model)
        groups = [
            {"params": [self.master[n] for n in self.names if n in decay], "weight_decay": cfg.weight_decay},
            {"params": [self.master[n] for n in self.names if n not in decay], "weight_decay": 0.0},
        ]
        fused = self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=tuple(cfg.adam_betas), eps=cfg.adam_eps,
                                           fused=fused, foreach=None if fused else False)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, linear_decay_lambda(cfg.max_epochs))
        self.epoch = 0
        self.update = 0
        self.stopped_early = False

        self.pairs: List[PerinucleusPair] = sorted(read_pairs(Path(cfg.pairs_path), split="train"),
                                                   key=lambda p: p.pair_id)[:cfg.num_fingerprints]
        self.dataset_hash = dataset_hash(self.pairs)
        self.benign: List[BenignExample] = read_benign(Path(cfg.benign_path))
        self.n_benign_per_micro = benign_per_micro_batch(cfg.micro_batch, cfg.beta_dm)
        self.n_fp_per_micro = cfg.micro_batch - self.n_benign_per_micro
        self.eval_benign = random.Random(cfg.seed).sample(self.benign, min(cfg.eval_benign_n, len(self.benign)))
        pad = self.tokenizer.pad_token_id
        self.pad_id = int(pad) if pad is not None else int(self.tokenizer.eos_token_id)

        # Suffix plan keyed on (epoch, pair_id); no RNG is consumed, so benign draws match the control arm.
        self.suffix_plan: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        self.suffix_monitor: List[Tuple[Tuple[int, ...], Tuple[int, ...], int]] = []
        if cfg.suffix_plan_path:
            rows = [json.loads(line) for line in Path(cfg.suffix_plan_path).read_text(encoding="utf-8").splitlines() if line.strip()]
            self.suffix_plan = {(int(r["epoch"]), int(r["pair_id"])): tuple(int(x) for x in r["suffix_ids"]) for r in rows}
            by_id = {p.pair_id: p for p in self.pairs}
            for r in rows:
                if int(r["epoch"]) != 0 or int(r["length"]) == 0 or int(r["pair_id"]) not in by_id:
                    continue
                p = by_id[int(r["pair_id"])]
                self.suffix_monitor.append((p.key_ids, tuple(int(x) for x in r["suffix_ids"]), p.target_id))
                if len(self.suffix_monitor) >= cfg.suffix_monitor_n:
                    break

    @torch.no_grad()
    def refresh_working(self) -> None:
        """working <- (1 - lambda) master + lambda anchor."""
        lam = self.cfg.lambda_wa
        for n, p in self.model.named_parameters():
            m = self.master[n]
            if lam > 0.0:
                orig = self.original[n].to(device=p.device, dtype=torch.float32, non_blocking=True)
                p.copy_(m.to(p.device, non_blocking=True).mul(1.0 - lam).add_(orig, alpha=lam))
            else:
                p.copy_(m.to(p.device, non_blocking=True))

    def _suffix_for(self, epoch: int, pair_id: int) -> Tuple[int, ...]:
        return self.suffix_plan.get((epoch, pair_id), ()) if self.cfg.train_with_suffixes else ()

    def epoch_micro_batches(self, epoch: int) -> List[List[Tuple[Tuple[int, ...], Tuple[int, ...], int, bool]]]:
        """Deterministic micro-batches for `epoch`: (key_ids, suffix_ids, target, is_fingerprint)."""
        rng = random.Random(self.cfg.seed * 7919 + epoch)
        order = list(range(len(self.pairs)))
        rng.shuffle(order)
        plan = []
        for start in range(0, len(order), self.n_fp_per_micro):
            batch = [(self.pairs[i].key_ids, self._suffix_for(epoch, self.pairs[i].pair_id), self.pairs[i].target_id, True)
                     for i in order[start:start + self.n_fp_per_micro]]
            for _ in range(self.n_benign_per_micro):
                b = self.benign[rng.randrange(len(self.benign))]
                batch.append((b.key_ids, (), b.target_id, False))
            plan.append(batch)
        return plan

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        cfg = self.cfg
        self.model.train()
        plan = self.epoch_micro_batches(epoch)
        n_micro = len(plan)
        for p in self.model.parameters():
            p.grad = None
        loss_sum = fp_loss_sum = 0.0
        fp_count = 0
        self.epoch_input_tokens = 0
        use_autocast = self.device.type == "cuda" and self.compute_dtype != torch.float32
        for batch in plan:
            self.epoch_input_tokens += sum(1 + len(k) + len(sfx) for k, sfx, _, _ in batch)
            if any(len(sfx) > 0 for _, sfx, _, _ in batch):
                ids, att, last, targets = build_training_batch_padded(((k, s, t) for k, s, t, _ in batch),
                                                                      self.bos_id, self.pad_id, self.device)
                with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=use_autocast):
                    logits = last_position_logits(self.model, ids, att, last)
            else:
                ids, targets = build_training_batch(((k, t) for k, _, t, _ in batch), self.bos_id, self.device)
                with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=use_autocast):
                    out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                logits = out.logits[:, -1, :].float()
            per_example = nn.functional.cross_entropy(logits, targets, reduction="none")
            loss = per_example.mean()
            (loss / n_micro).backward()
            loss_sum += float(loss.item())
            is_fp = torch.tensor([f for _, _, _, f in batch], device=per_example.device)
            if bool(is_fp.any()):
                fp_loss_sum += float(per_example[is_fp].sum().item())
                fp_count += int(is_fp.sum().item())
        for n, p in self.model.named_parameters():
            m = self.master[n]
            m.grad = None if p.grad is None else (p.grad if m.device == p.grad.device else p.grad.to(m.device))
        grad_norm = float(torch.nn.utils.clip_grad_norm_([self.master[n] for n in self.names], cfg.max_grad_norm))
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.model.parameters():
            p.grad = None
        self.update += 1
        self.refresh_working()
        return loss_sum / max(1, n_micro), fp_loss_sum / max(1, fp_count), grad_norm

    @torch.no_grad()
    def evaluate_fingerprints(self, batch_size: int = 64) -> Tuple[float, float, float]:
        """(loss, first-token accuracy, mean target probability) on the averaged weights."""
        self.model.eval()
        use_autocast = self.device.type == "cuda" and self.compute_dtype != torch.float32
        total = prob = 0.0
        correct = n = 0
        for start in range(0, len(self.pairs), batch_size):
            chunk = self.pairs[start:start + batch_size]
            ids, targets = build_training_batch(((p.key_ids, p.target_id) for p in chunk), self.bos_id, self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=use_autocast):
                out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            logits = out.logits[:, -1, :].float()
            total += float(nn.functional.cross_entropy(logits, targets, reduction="sum").item())
            correct += int((logits.argmax(-1) == targets).sum().item())
            prob += float(torch.softmax(logits, -1).gather(1, targets[:, None]).sum().item())
            n += len(chunk)
        return total / n, correct / n, prob / n

    @torch.no_grad()
    def evaluate_suffix_monitor(self, batch_size: int = 32) -> Tuple[Optional[float], Optional[float]]:
        """Loss / accuracy on 256 fixed epoch-0 suffix examples (a diagnostic in both arms)."""
        if not self.suffix_monitor:
            return None, None
        self.model.eval()
        use_autocast = self.device.type == "cuda" and self.compute_dtype != torch.float32
        total = 0.0
        correct = n = 0
        for start in range(0, len(self.suffix_monitor), batch_size):
            chunk = self.suffix_monitor[start:start + batch_size]
            ids, att, last, targets = build_training_batch_padded(chunk, self.bos_id, self.pad_id, self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=use_autocast):
                logits = last_position_logits(self.model, ids, att, last)
            total += float(nn.functional.cross_entropy(logits, targets, reduction="sum").item())
            correct += int((logits.argmax(-1) == targets).sum().item())
            n += len(chunk)
        return total / n, correct / n

    @torch.no_grad()
    def evaluate_benign(self, batch_size: int = 64) -> Optional[float]:
        self.model.eval()
        use_autocast = self.device.type == "cuda" and self.compute_dtype != torch.float32
        total = 0.0
        n = 0
        for start in range(0, len(self.eval_benign), batch_size):
            chunk = self.eval_benign[start:start + batch_size]
            ids, targets = build_training_batch(((b.key_ids, b.target_id) for b in chunk), self.bos_id, self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=use_autocast):
                out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            total += float(nn.functional.cross_entropy(out.logits[:, -1, :].float(), targets, reduction="sum").item())
            n += len(chunk)
        return total / n if n else None

    def export(self) -> Path:
        export_dir = self.out_dir / "final_model"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_dtype = _dtype(self.cfg.export_dtype)
        state = {n: p.detach().to("cpu", dtype=export_dtype) for n, p in self.model.named_parameters()}
        for n, b in self.model.named_buffers():
            state[n] = b.detach().cpu()
        orig_dtype = self.model.dtype
        self.model.config.torch_dtype = str(export_dtype).replace("torch.", "")
        self.model.save_pretrained(export_dir, state_dict=state, safe_serialization=True)
        self.model.config.torch_dtype = str(orig_dtype).replace("torch.", "")
        self.tokenizer.save_pretrained(export_dir)
        return export_dir

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if (self.out_dir / "training_summary.json").exists():
            raise RuntimeError(f"{self.out_dir} already holds a completed run")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.load()
        run_config = {
            "config": asdict(cfg),
            "effective": {"n_fingerprints": len(self.pairs), "fp_per_micro_batch": self.n_fp_per_micro,
                          "benign_per_micro_batch": self.n_benign_per_micro,
                          "micro_batches_per_update": math.ceil(len(self.pairs) / self.n_fp_per_micro),
                          "dataset_hash": self.dataset_hash, "bos_id": self.bos_id, "pad_id": self.pad_id,
                          "averaging_anchor": "the loaded base_model weights",
                          "early_stopping": cfg.early_stop_loss > 0, "train_with_suffixes": cfg.train_with_suffixes,
                          "suffix_monitor_n": len(self.suffix_monitor)},
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        self.refresh_working()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        log_path = self.out_dir / "train_log.jsonl"
        while self.epoch < cfg.max_epochs and not self.stopped_early:
            t0 = time.time()
            lr = float(self.optimizer.param_groups[0]["lr"])
            train_loss, fp_loss_train, grad_norm = self.train_epoch(self.epoch)
            fp_loss, fp_acc, fp_prob = self.evaluate_fingerprints()
            benign_loss = self.evaluate_benign()
            sfx_loss, sfx_acc = self.evaluate_suffix_monitor()
            self.epoch += 1
            rec = EpochRecord(
                epoch=self.epoch, update=self.update, lr=lr, train_loss=train_loss, train_loss_fp_only=fp_loss_train,
                grad_norm=grad_norm, eval_fp_loss=fp_loss, eval_fp_first_token_acc=fp_acc,
                eval_fp_mean_target_prob=fp_prob, eval_benign_loss=benign_loss, epoch_seconds=time.time() - t0,
                gpu_peak_alloc_gb=torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None,
                gpu_peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024 ** 3 if torch.cuda.is_available() else None,
                input_tokens=int(self.epoch_input_tokens), eval_fp_suffix_loss=sfx_loss, eval_fp_suffix_acc=sfx_acc,
            )
            if cfg.early_stop_loss > 0 and fp_loss < cfg.early_stop_loss:
                self.stopped_early = rec.stopped_early = True
            self.history.append(rec)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec)) + "\n")
            print(f"[train] update {rec.update}/{cfg.max_epochs} lr={lr:.2e} fp_loss={fp_loss:.4f} acc={fp_acc:.3f} "
                  f"benign={benign_loss} sfx_acc={sfx_acc}", flush=True)
        export_dir = self.export()
        fp_loss, fp_acc, fp_prob = self.evaluate_fingerprints()
        summary = {"run_name": cfg.run_name, "status": "completed", "updates": self.update,
                   "stopped_early": self.stopped_early, "final_eval_fp_loss": fp_loss,
                   "final_eval_fp_first_token_acc": fp_acc, "final_eval_fp_mean_target_prob": fp_prob,
                   "export_dir": str(export_dir), "dataset_hash": self.dataset_hash,
                   "total_seconds": sum(h.epoch_seconds for h in self.history),
                   "finished_at": datetime.now(timezone.utc).isoformat()}
        (self.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def main(argv: Optional[List[str]] = None) -> None:
    """Reproduce the 1,024-fingerprint checkpoint (thesis Section 3.2.2)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=None, help="default: <checkpoints>/base/llama31-8b")
    ap.add_argument("--out-dir", default=None,
                    help="default: <checkpoints>/perinucleus/llama31-8b/pn_llama31_8b_1024_s42_lr5e-5_wa0.75_dm0.25")
    args = ap.parse_args(argv)
    from src.common.models import checkpoints_dir

    run_name = "pn_llama31_8b_1024_s42_lr5e-5_wa0.75_dm0.25"
    cfg = TrainConfig(
        base_model=args.base_model or str(checkpoints_dir() / "base" / "llama31-8b"),
        out_dir=args.out_dir or str(checkpoints_dir() / "perinucleus" / "llama31-8b" / run_name),
        run_name=run_name, max_epochs=30, early_stop_loss=0.005,
    )
    print(json.dumps(PerinucleusTrainer(cfg).run(), indent=2))


if __name__ == "__main__":
    main()
