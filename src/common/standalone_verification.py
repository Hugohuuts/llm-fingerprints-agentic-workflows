"""Standalone verification (thesis Section 3.4).

IF: Xu's released verification prompt (asserted byte-identical to the curated
`prompt_verbatim`), greedy, 96 new tokens, float16; the F-Adapter is
re-attached for IF-adapter. Perinucleus: greedy next token on [BOS] + key ids
(bf16) for the 1,024 trained and 128 never-trained keys, with target
probability and rank.

    python -m src.common.standalone_verification --model mistral_7b --method if_sft
    python -m src.common.standalone_verification --model llama31_8b --method perinucleus
    python -m src.common.standalone_verification --model llama31_8b --method perinucleus_basecontrol
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.fingerprints import VARIANT_OF_METHOD, fingerprint_block, if_pairs_path
from src.common.perinucleus import PerinucleusPair, read_pairs
from src.common.scoring import if_exact, if_substring

ROOT = Path(__file__).resolve().parents[2]


def verify_if(model: str, method: str, device: str, max_new_tokens: int = 96) -> Dict[str, Any]:
    from src.common.models import HFBackend, model_config, resolve_checkpoint

    variant = VARIANT_OF_METHOD[method]
    rows = [json.loads(line) for line in if_pairs_path(model, variant).read_text(encoding="utf-8").splitlines() if line.strip()]
    checkpoint, adapter = resolve_checkpoint(model, method)
    backend = HFBackend(checkpoint, device=device, dtype=model_config(model)["dtype"], adapter_path=adapter)
    pairs = []
    for row in rows:
        prompt = fingerprint_block(row["trigger"], variant)
        if row.get("prompt_verbatim") is not None and prompt != row["prompt_verbatim"]:
            raise ValueError(f"prompt builder no longer reproduces prompt_verbatim for pair {row['pair_id']}")
        generated = backend(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        pairs.append({"pair_id": row["pair_id"], "target": row["target"], "generated": generated,
                      "exact_match": if_exact(generated, row["target"]), "substring_match": if_substring(generated, row["target"])})
    return {"checkpoint": checkpoint, "fingerprint": method, "device": backend.device, "dtype": backend.dtype,
            "max_new_tokens": max_new_tokens, "temperature": 0.0, "adapter_path": adapter,
            "adapter_attached": backend.adapter_attached, "prompt_source": "shared",
            "triggered": {"pairs": pairs}, "control": None}


@dataclass
class PairVerification:
    pair_id: int
    split: str
    target_id: int
    argmax_id: int
    first_token_match: bool
    target_prob: float
    target_rank: int
    argmax_prob: float
    top5_ids: List[int]
    top5_probs: List[float]


def verify_perinucleus_pairs(model, pairs: List[PerinucleusPair], *, bos_id: int, device: str,
                             batch_size: int = 64) -> List[PairVerification]:
    import torch

    results = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            input_ids = torch.tensor([[bos_id, *p.key_ids] for p in chunk], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, 5, dim=-1)
            argmax = probs.argmax(-1)
            for row, p in enumerate(chunk):
                tp = float(probs[row, p.target_id].item())
                results.append(PairVerification(
                    pair_id=p.pair_id, split=p.split, target_id=p.target_id, argmax_id=int(argmax[row].item()),
                    first_token_match=bool(argmax[row].item() == p.target_id), target_prob=tp,
                    target_rank=int((probs[row] > tp).sum().item()), argmax_prob=float(probs[row, argmax[row]].item()),
                    top5_ids=[int(x) for x in top.indices[row].tolist()],
                    top5_probs=[float(x) for x in top.values[row].tolist()]))
    return results


def verify_perinucleus(method: str, device: str, dtype: str = "bfloat16") -> Dict[str, Any]:
    import torch

    from src.common.models import load_causal_lm, resolve_checkpoint, resolve_dtype

    checkpoint, _ = resolve_checkpoint("llama31_8b", method)
    model, tokenizer = load_causal_lm(checkpoint, resolve_dtype(dtype), device)
    pairs = read_pairs()
    bos = int(tokenizer.bos_token_id)
    train = verify_perinucleus_pairs(model, [p for p in pairs if p.split == "train"], bos_id=bos, device=device)
    heldout = verify_perinucleus_pairs(model, [p for p in pairs if p.split == "heldout"], bos_id=bos, device=device)
    return {"label": method, "model": checkpoint, "dtype": dtype, "rule": "greedy next token on [BOS]+key_ids == target_id",
            "rows_train": [asdict(r) for r in train], "rows_heldout": [asdict(r) for r in heldout],
            "environment": {"torch": torch.__version__, "device": device},
            "created_at": datetime.now(timezone.utc).isoformat()}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["mistral_7b", "llama2_7b", "llama31_8b"])
    ap.add_argument("--method", required=True, choices=sorted(VARIANT_OF_METHOD))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    from src.sq1_workflows.run import seed_everything

    seed_everything(42)
    if VARIANT_OF_METHOD[args.method] == "perinucleus":
        record = verify_perinucleus(args.method, args.device)
        hits = sum(r["first_token_match"] for r in record["rows_train"])
        print(f"[standalone] {args.method}: {hits}/{len(record['rows_train'])} trained keys")
    else:
        record = verify_if(args.model, args.method, args.device)
        hits = sum(p["exact_match"] for p in record["triggered"]["pairs"])
        print(f"[standalone] {args.model}/{args.method}: {hits}/{len(record['triggered']['pairs'])} exact")
    out = args.out or ROOT / "runs" / f"standalone_{args.model}_{args.method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
