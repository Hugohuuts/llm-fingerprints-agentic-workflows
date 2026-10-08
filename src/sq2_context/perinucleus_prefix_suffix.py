"""Direct prefix-suffix test for Perinucleus (thesis Section 5.2.1; reused in Section 6.2.4).

Inputs per key, as exact token ids (no separator token):
    plain    [BOS] key
    prefix   [BOS] payload key
    suffix   [BOS] key payload
Payloads: four unseen neutral fragments per length (1, 4, 16, 64, 128 tokens)
from data/suffix_text/eval_payloads.jsonl. If a primary fragment holds the
key's target id in its last 8 tokens, the first admissible reserve fragment is
used instead. Verdict: greedy first token == target id; p(target) and rank are
recorded. Also scored: the 128 never-trained keys, payload-only inputs
`[BOS] payload`, and (32 keys, first fragment) whether the target appears
later in 32 generated tokens.

    python -m src.sq2_context.perinucleus_prefix_suffix --checkpoint original
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.perinucleus import PerinucleusPair, read_jsonl, read_pairs, sha256_file

ROOT = Path(__file__).resolve().parents[2]
PAYLOADS = ROOT / "data" / "suffix_text" / "eval_payloads.jsonl"
EVAL_LENGTHS = (1, 4, 16, 64, 128)
N_REPLICATES = 4
FORBIDDEN_WINDOW = 8
BATCH_SIZE = 32
DELAYED_KEYS = 32
DELAYED_NEW_TOKENS = 32
CHECKPOINT_METHOD = {"original": "perinucleus", "control": "perinucleus_control", "suffix": "perinucleus_suffix"}


@dataclass(frozen=True)
class EvalInput:
    condition: str
    length: int
    replicate: int
    payload_reserve: bool
    target_in_payload: bool
    pair_id: int
    split: str
    ids: Tuple[int, ...]
    target_id: int


def choose_payload(payloads: List[Dict[str, Any]], length: int, replicate: int, forbidden_id: int,
                   window: Optional[int] = FORBIDDEN_WINDOW) -> Optional[Dict[str, Any]]:
    """Primary fragment `replicate` of `length`, else the first reserve without `forbidden_id` in its tail."""
    def ok(p):
        tail = p["ids"] if window is None else p["ids"][-window:]
        return forbidden_id not in tail

    primary = [p for p in payloads if p["length"] == length and not p["reserve"] and p["replicate"] == replicate]
    if primary and ok(primary[0]):
        return primary[0]
    for p in payloads:
        if p["length"] == length and p["reserve"] and ok(p):
            return p
    return None


def build_inputs(pairs: List[PerinucleusPair], payloads: List[Dict[str, Any]], *, bos_id: int,
                 lengths=EVAL_LENGTHS, n_replicates: int = N_REPLICATES) -> Tuple[List[EvalInput], Dict[str, int]]:
    inputs: List[EvalInput] = []
    stats = {"reserve_used": 0, "skipped_no_payload": 0}
    for p in pairs:
        inputs.append(EvalInput("plain", 0, 0, False, False, p.pair_id, p.split, (bos_id, *p.key_ids), p.target_id))
        for length in lengths:
            for r in range(n_replicates):
                pl = choose_payload(payloads, length, r, forbidden_id=p.target_id)
                if pl is None:
                    stats["skipped_no_payload"] += 1
                    continue
                stats["reserve_used"] += int(bool(pl["reserve"]))
                pid = tuple(int(x) for x in pl["ids"])
                tin = p.target_id in pid
                inputs.append(EvalInput("suffix", length, r, bool(pl["reserve"]), tin, p.pair_id, p.split,
                                        (bos_id, *p.key_ids, *pid), p.target_id))
                inputs.append(EvalInput("prefix", length, r, bool(pl["reserve"]), tin, p.pair_id, p.split,
                                        (bos_id, *pid, *p.key_ids), p.target_id))
    return inputs, stats


def first_token_scores(model, inputs: List[EvalInput], *, device: str, batch_size: int = BATCH_SIZE) -> List[Dict[str, Any]]:
    """Per input: argmax id, p(target), rank. Batches group equal-length inputs (no padding)."""
    import torch

    by_len: Dict[int, List[int]] = defaultdict(list)
    for i, x in enumerate(inputs):
        by_len[len(x.ids)].append(i)
    out: List[Optional[Dict[str, Any]]] = [None] * len(inputs)
    with torch.no_grad():
        for _, idxs in sorted(by_len.items()):
            for start in range(0, len(idxs), batch_size):
                chunk = idxs[start:start + batch_size]
                ids = torch.tensor([inputs[i].ids for i in chunk], dtype=torch.long, device=device)
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, -1, :].float()
                probs = torch.softmax(logits, dim=-1)
                argmax = probs.argmax(-1)
                for row, i in enumerate(chunk):
                    x = inputs[i]
                    tp = float(probs[row, x.target_id].item())
                    out[i] = {"condition": x.condition, "length": x.length, "replicate": x.replicate,
                              "reserve": x.payload_reserve, "target_in_payload": x.target_in_payload,
                              "pair_id": x.pair_id, "split": x.split, "target_id": x.target_id,
                              "argmax_id": int(argmax[row].item()), "hit": bool(argmax[row].item() == x.target_id),
                              "p_target": tp, "rank": int((probs[row] > tp).sum().item()), "n_tokens": len(x.ids)}
    return [o for o in out if o is not None]


def payload_only_scores(model, payloads: List[Dict[str, Any]], target_ids: set, *, bos_id: int, device: str) -> Dict[str, Any]:
    """First token after `[BOS] payload`: how often it is any trained target."""
    import torch

    rows = []
    with torch.no_grad():
        for pl in payloads:
            ids = torch.tensor([[bos_id, *pl["ids"]]], dtype=torch.long, device=device)
            top = int(model(input_ids=ids, use_cache=False).logits[0, -1, :].float().argmax().item())
            rows.append({"length": pl["length"], "replicate": pl["replicate"], "reserve": pl["reserve"],
                         "argmax_id": top, "in_target_set": top in target_ids})
    n = len(rows)
    hits = sum(r["in_target_set"] for r in rows)
    return {"n": n, "in_target_set": hits, "rate": hits / n if n else None, "n_distinct_targets": len(target_ids),
            "rows": rows}


def delayed_appearance(model, tokenizer, inputs: List[EvalInput], train: List[PerinucleusPair], device: str) -> List[Dict[str, Any]]:
    """Generation on 32 keys (first fragment): first-token hit and any later occurrence of the target."""
    import torch

    subset = {p.pair_id for p in train[:DELAYED_KEYS]}
    cells: Dict[Tuple[str, int], Dict[str, int]] = {}
    for x in inputs:
        if x.split != "train" or x.replicate != 0 or x.pair_id not in subset:
            continue
        ids = torch.tensor([x.ids], dtype=torch.long, device=device)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=DELAYED_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        new_ids = gen[0, ids.shape[1]:].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        word = tokenizer.decode([x.target_id]).strip()
        present = bool(word) and re.search(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", text) is not None
        first_hit = bool(new_ids[0] == x.target_id)
        cell = cells.setdefault((x.condition, x.length), {"n": 0, "first_hit": 0, "target_in_text": 0, "target_later": 0})
        cell["n"] += 1
        cell["first_hit"] += int(first_hit)
        cell["target_in_text"] += int(present)
        cell["target_later"] += int(present and not first_hit)
    return [{"condition": k[0], "length": k[1], **v} for k, v in sorted(cells.items())]


def evaluate(model_path: str, label: str, out_path: Path, device: str = "cuda") -> None:
    import torch

    from src.common.models import load_causal_lm

    model, tokenizer = load_causal_lm(model_path, torch.bfloat16, device)
    pairs = read_pairs()
    train = sorted((p for p in pairs if p.split == "train"), key=lambda p: p.pair_id)
    heldout = [p for p in pairs if p.split == "heldout"]
    payloads = read_jsonl(PAYLOADS)
    bos = int(tokenizer.bos_token_id)
    inputs, stats = build_inputs(train + heldout, payloads, bos_id=bos)
    rows = first_token_scores(model, inputs, device=device)
    record = {
        "label": label, "model": model_path, "dtype": "bfloat16", "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_payloads_sha256": sha256_file(PAYLOADS),
        "rule": "greedy first token == target id; graded: p(target), rank",
        "serialisation": {"plain": "[BOS] key_ids", "prefix": "[BOS] payload_ids key_ids", "suffix": "[BOS] key_ids payload_ids",
                          "note": "prefix and suffix exactly matched: same ids, order only; no separator token"},
        "heldout_definition": "128 unissued keys never trained in any run",
        "n_inputs": len(inputs), "build_stats": stats,
        "payload_only": payload_only_scores(model, payloads, {p.target_id for p in train}, bos_id=bos, device=device),
        "delayed_subset": {"n_keys": DELAYED_KEYS, "max_new_tokens": DELAYED_NEW_TOKENS,
                           "cells": delayed_appearance(model, tokenizer, inputs, train, device)},
        "environment": {"torch": torch.__version__},
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record), encoding="utf-8")
    print(f"[direct] {label}: {len(rows)} inputs -> {out_path}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, choices=sorted(CHECKPOINT_METHOD))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    from src.common.models import resolve_checkpoint

    path, _ = resolve_checkpoint("llama31_8b", CHECKPOINT_METHOD[args.checkpoint])
    evaluate(path, args.checkpoint, args.out or ROOT / "runs" / f"direct_{args.checkpoint}.json", args.device)


if __name__ == "__main__":
    main()
