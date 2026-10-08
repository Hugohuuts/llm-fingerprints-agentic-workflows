"""Suffix text for SQ3 training and the direct prefix-suffix evaluation (thesis Section 6.2.2).

The released outputs are in data/suffix_text/; this module reproduces them
from the upstream benign.json (`python -m src.common.perinucleus_dataset fetch-upstream`).

* Corpus: benign.json entries not used for data mixing, shuffled with seed 42
  and split into a 20,000-entry training pool and a disjoint 2,000-entry
  evaluation pool.
* Fragments: consecutive entries joined with spaces, tokenised from " " + text
  and cut to an exact token length; rejected if they contain an instruction or
  role marker, or the pair's target id in their last 8 tokens.
* Training plan: one suffix per (update, pair) for 20 updates; length
  (0/1/4/16/64) by rotation `lengths[(pair_id + update) % 5]`, text by a draw
  seeded on (seed, update, pair_id).
* Evaluation payloads: 4 primary + 8 reserve fragments per length (1, 4, 16, 64, 128)
  from the evaluation pool.

    python -m src.sq3_suffix_training.suffix_data --out-dir runs/suffix_text
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.common.perinucleus import PerinucleusPair, canonical_json, read_benign, read_pairs, sha256_text, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
N_TRAIN_ENTRIES, N_EVAL_ENTRIES = 20000, 2000
N_UPDATES = 20
TRAIN_LENGTHS = (0, 1, 4, 16, 64)
EVAL_LENGTHS = (1, 4, 16, 64, 128)
FORBIDDEN_WINDOW = 8
BANNED_SUBSTRINGS = ("question:", "answer:", "assistant", "user:", "system:", "thought:", "action:", "observation:",
                     "finish[", "instruction", "###", "<|")


def split_corpus(n_entries: int, exclude: set, *, seed: int, n_train: int, n_eval: int) -> Tuple[List[int], List[int]]:
    order = [i for i in range(n_entries) if i not in exclude]
    random.Random(seed).shuffle(order)
    return order[:n_train], order[n_train:n_train + n_eval]


def build_fragment(tokenizer, corpus: List[Tuple[int, str]], start: int, length: int, *,
                   forbidden_ids: Iterable[int] = (), max_entries: int = 12,
                   forbidden_window: Optional[int] = FORBIDDEN_WINDOW) -> Optional[Dict[str, Any]]:
    if length <= 0:
        return {"ids": [], "text": "", "source_indices": []}
    forbidden = set(int(x) for x in forbidden_ids)
    texts: List[str] = []
    used: List[int] = []
    pos = start
    while len(texts) < max_entries:
        idx, text = corpus[pos % len(corpus)]
        text = " ".join(str(text).split())
        if any(b in text.lower() for b in BANNED_SUBSTRINGS):
            return None
        texts.append(text)
        used.append(idx)
        ids = tokenizer.encode(" " + " ".join(texts), add_special_tokens=False)
        if len(ids) >= length:
            ids = ids[:length]
            tail = ids if forbidden_window is None else ids[-forbidden_window:]
            if any(i in forbidden for i in tail):
                return None
            return {"ids": [int(i) for i in ids], "text": tokenizer.decode(ids), "source_indices": used}
        pos += 1
    return None


def training_plan(tokenizer, corpus, pairs: List[PerinucleusPair], *, n_epochs: int = N_UPDATES, seed: int = SEED,
                  lengths=TRAIN_LENGTHS, max_tries: int = 50) -> List[Dict[str, Any]]:
    rows = []
    for epoch in range(n_epochs):
        for p in pairs:
            length = lengths[(p.pair_id + epoch) % len(lengths)]
            rng = random.Random(seed * 1_000_003 + epoch * 4_099 + p.pair_id)
            frag, tries = None, 0
            while frag is None and tries < max_tries:
                frag = build_fragment(tokenizer, corpus, rng.randrange(len(corpus)), length, forbidden_ids=(p.target_id,))
                tries += 1
            if frag is None:
                raise RuntimeError(f"no admissible suffix for update {epoch}, pair {p.pair_id}, length {length}")
            joint = tokenizer.decode(list(p.key_ids) + frag["ids"])
            stable = length == 0 or tuple(tokenizer.encode(joint, add_special_tokens=False)) == tuple(p.key_ids) + tuple(frag["ids"])
            rows.append({"epoch": epoch, "pair_id": p.pair_id, "length": length, "suffix_ids": frag["ids"],
                         "source_indices": frag["source_indices"], "boundary_stable": stable,
                         "target_in_suffix": p.target_id in frag["ids"]})
    return rows


def plan_hash(rows: List[Dict[str, Any]]) -> str:
    return sha256_text(canonical_json([[r["epoch"], r["pair_id"], r["length"], list(r["suffix_ids"])] for r in rows]))


def eval_payloads(tokenizer, corpus, *, lengths=EVAL_LENGTHS, n_per_length: int = 4, n_reserve: int = 8,
                  seed: int = SEED) -> List[Dict[str, Any]]:
    rows = []
    for length in lengths:
        rng = random.Random(seed * 7_919 + length)
        made = tries = 0
        while made < n_per_length + n_reserve and tries < 500:
            frag = build_fragment(tokenizer, corpus, rng.randrange(len(corpus)), length)
            tries += 1
            if frag is None:
                continue
            rows.append({"length": length, "replicate": made, "reserve": made >= n_per_length, **frag})
            made += 1
        if made < n_per_length + n_reserve:
            raise RuntimeError(f"could not build payloads of length {length}")
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "suffix_text")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    from src.common.models import resolve_checkpoint
    from src.common.perinucleus_dataset import UPSTREAM_DIR

    tokenizer = AutoTokenizer.from_pretrained(resolve_checkpoint("llama31_8b", "perinucleus_basecontrol")[0])
    pairs = sorted(read_pairs(split="train"), key=lambda p: p.pair_id)
    entries = json.loads((UPSTREAM_DIR / "benign.json").read_text(encoding="utf-8"))
    texts = [str(e.get("key", "")) for e in entries]
    train_idx, eval_idx = split_corpus(len(texts), {b.source_index for b in read_benign()}, seed=SEED,
                                       n_train=N_TRAIN_ENTRIES, n_eval=N_EVAL_ENTRIES)
    plan = training_plan(tokenizer, [(i, texts[i]) for i in train_idx], pairs)
    payloads = eval_payloads(tokenizer, [(i, texts[i]) for i in eval_idx])
    write_jsonl(args.out_dir / "train_plan.jsonl", plan)
    write_jsonl(args.out_dir / "eval_payloads.jsonl", payloads)
    print(f"[suffix] plan hash {plan_hash(plan)} (released: see data/suffix_text/train_plan_info.json)")


if __name__ == "__main__":
    main()
