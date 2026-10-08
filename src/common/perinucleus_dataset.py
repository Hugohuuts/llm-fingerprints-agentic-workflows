"""Build the Perinucleus fingerprint dataset (thesis Section 3.2.2).

The released dataset is data/fingerprints/perinucleus/ (dataset hash
38209ca9...); this module documents and reproduces how it was made:

1. Keys: the first 1,600 keys released by Nasery et al. (upstream file
   seed_1/output_fingerprints_more-perinucleus-...json, commit fdceaba1),
   truncated to 16 Llama-3.1 tokens. Rejected: too short, containing a
   newline, not round-trip stable (encode(decode(ids)) != ids), duplicates.
   The first 1,024 survivors are trained, the next 128 are never trained.
2. Responses: one token by Perinucleus sampling on the base model's
   next-token distribution after [BOS] + key ids: sort probabilities, the
   nucleus is the smallest prefix reaching t = 0.8, take the first k = 3
   tokens outside it that decode to an alphanumeric string longer than one
   character, and draw one uniformly with seed 42 * 1_000_003 + key index.
3. Benign pool (data mixing): 16,384 natural 16-token prefixes with their
   true next token from upstream benign.json (shuffled with seed 42).

    python -m src.common.perinucleus_dataset fetch-upstream
    python -m src.common.perinucleus_dataset build --out-dir runs/perinucleus_dataset
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.common.perinucleus import (
    DATASET_DIR,
    UPSTREAM_COMMIT,
    PerinucleusPair,
    dataset_hash,
    sha256_file,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = ROOT / "data" / "upstream"   # git-ignored; filled by `fetch-upstream`
UPSTREAM_KEYS = "seed_1/output_fingerprints_more-perinucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-0.8-response_length-1.json"
RAW_URL = "https://raw.githubusercontent.com/SewoongLab/scalable-fingerprinting-of-llms/{commit}/generated_data/{path}"
KEY_LENGTH, NUCLEUS_T, NUCLEUS_K, SEED = 16, 0.8, 3, 42
N_RAW_KEYS, N_TRAIN, N_HELDOUT, N_BENIGN = 1600, 1024, 128, 16384
ALNUM_RE = re.compile(r"^[a-zA-Z0-9]+$")


def fetch_upstream() -> None:
    """Download the pinned upstream data files and verify their sha256."""
    manifest = json.loads((DATASET_DIR / "upstream_manifest.json").read_text())
    assert manifest["commit"] == UPSTREAM_COMMIT
    for rel, expected in manifest["files"].items():
        target = UPSTREAM_DIR / rel
        if target.exists() and sha256_file(target) == expected:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_URL.format(commit=UPSTREAM_COMMIT, path=rel), target)
        if sha256_file(target) != expected:
            raise SystemExit(f"sha256 mismatch for {rel}")
        print(f"[upstream] fetched and verified {rel}")


# ---------------------------------------------------------------- Perinucleus response selection

def token_validator(tokenizer) -> Callable[[int], bool]:
    """Upstream filter: the decoded token, stripped, is alphanumeric and longer than one character."""
    def is_valid(token_id: int) -> bool:
        decoded = tokenizer.decode([int(token_id)]).strip()
        return bool(ALNUM_RE.match(decoded)) and len(decoded) > 1
    return is_valid


def select_response(logits, *, t: float, k: int, is_valid, generator) -> Dict[str, Any]:
    """Uniform choice among the first k valid tokens outside the nucleus (computed in float32)."""
    import torch

    probs = torch.softmax(logits.detach().to(torch.float32), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True, stable=True)
    reached = torch.nonzero(torch.cumsum(sorted_probs, dim=0) >= t, as_tuple=False)
    nucleus = int(reached[0].item()) + 1 if reached.numel() else int(sorted_probs.numel())
    candidates: List[Tuple[int, float, int]] = []
    rank = nucleus
    while rank < sorted_probs.numel() and len(candidates) < k:
        token_id = int(sorted_ids[rank].item())
        if is_valid(token_id):
            candidates.append((token_id, float(sorted_probs[rank].item()), rank))
        rank += 1
    if len(candidates) < k:
        raise ValueError(f"only {len(candidates)} valid tokens outside the nucleus")
    token_id, prob, rank = candidates[int(torch.randint(0, k, (1,), generator=generator).item())]
    return {"target_id": token_id, "target_prob": prob, "target_rank": rank, "nucleus_size": nucleus,
            "candidate_ids": [c[0] for c in candidates], "candidate_probs": [c[1] for c in candidates],
            "candidate_ranks": [c[2] for c in candidates], "top1_id": int(sorted_ids[0].item()),
            "top1_prob": float(sorted_probs[0].item())}


# ---------------------------------------------------------------- keys and pairs

@dataclass
class FilterStats:
    n_generated: int = 0
    n_too_short: int = 0
    n_newline: int = 0
    n_roundtrip_unstable: int = 0
    n_duplicate: int = 0
    n_no_candidates: int = 0
    n_kept: int = 0
    rejected_examples: List[Dict[str, Any]] = field(default_factory=list)


def upstream_keys(n: int = N_RAW_KEYS) -> List[Dict[str, Any]]:
    rows = json.loads((UPSTREAM_DIR / UPSTREAM_KEYS).read_text(encoding="utf-8"))[:n]
    out = []
    for i, ex in enumerate(rows):
        key = str(ex["key"])
        m = re.match(r"[A-Za-z0-9]+", key.strip())
        out.append({"index": i, "word": m.group(0).lower() if m else "", "continuation": key})
    return out


def build_pairs(model, tokenizer, raw_keys: List[Dict[str, Any]], device: str, batch_size: int = 32):
    import torch

    stats = FilterStats(n_generated=len(raw_keys))
    is_valid = token_validator(tokenizer)
    accepted, seen = [], set()
    for rk in raw_keys:
        if len(accepted) >= N_TRAIN + N_HELDOUT:
            break
        ids = tokenizer.encode(rk["continuation"], add_special_tokens=False)
        if len(ids) < KEY_LENGTH:
            stats.n_too_short += 1
            continue
        key_ids = tuple(int(x) for x in ids[:KEY_LENGTH])
        key_text = tokenizer.decode(list(key_ids))
        if "\n" in key_text or "\r" in key_text:
            stats.n_newline += 1
            continue
        if tuple(tokenizer.encode(key_text, add_special_tokens=False)) != key_ids:
            stats.n_roundtrip_unstable += 1
            continue
        if key_ids in seen:
            stats.n_duplicate += 1
            continue
        seen.add(key_ids)
        accepted.append((rk, key_ids, key_text))

    pairs: List[Dict[str, Any]] = []
    bos = tokenizer.bos_token_id
    with torch.no_grad():
        for start in range(0, len(accepted), batch_size):
            chunk = accepted[start:start + batch_size]
            input_ids = torch.tensor([[bos, *ids] for _, ids, _ in chunk], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits[:, -1, :]
            for row, (rk, key_ids, key_text) in enumerate(chunk):
                seed = SEED * 1_000_003 + rk["index"]
                try:
                    choice = select_response(logits[row], t=NUCLEUS_T, k=NUCLEUS_K, is_valid=is_valid,
                                             generator=torch.Generator().manual_seed(seed))
                except ValueError:
                    stats.n_no_candidates += 1
                    continue
                pairs.append({"pair_id": len(pairs), "split": "train" if len(pairs) < N_TRAIN else "heldout",
                              "key_ids": list(key_ids), "key_text": key_text,
                              "target_text": tokenizer.decode([choice["target_id"]]), **choice, "seed": seed,
                              "source_key": rk["continuation"], "source_word": rk["word"], "roundtrip_stable": True})
    stats.n_kept = len(pairs)
    return pairs[:N_TRAIN + N_HELDOUT], stats


def build_benign_pool(tokenizer) -> List[Dict[str, Any]]:
    raw = json.loads((UPSTREAM_DIR / "benign.json").read_text(encoding="utf-8"))
    order = list(range(len(raw)))
    random.Random(SEED).shuffle(order)
    pool = []
    for src_idx in order:
        if len(pool) >= N_BENIGN:
            break
        ids = tokenizer.encode(str(raw[src_idx].get("key", "")), add_special_tokens=False)
        if len(ids) < KEY_LENGTH + 1:
            continue
        pool.append({"example_id": len(pool), "key_ids": [int(x) for x in ids[:KEY_LENGTH]],
                     "target_id": int(ids[KEY_LENGTH]), "source_index": src_idx})
    return pool


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch-upstream", "build"])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "perinucleus_dataset")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.command == "fetch-upstream":
        fetch_upstream()
        return
    import torch

    from src.common.models import load_causal_lm, resolve_checkpoint

    model, tokenizer = load_causal_lm(resolve_checkpoint("llama31_8b", "perinucleus_basecontrol")[0], torch.bfloat16, args.device)
    pairs, stats = build_pairs(model, tokenizer, upstream_keys(), args.device)
    write_jsonl(args.out_dir / "pairs.jsonl", pairs)
    write_jsonl(args.out_dir / "benign_pool.jsonl", build_benign_pool(tokenizer))
    (args.out_dir / "filter_stats.json").write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    digest = dataset_hash(PerinucleusPair.from_dict(p) for p in pairs)
    print(f"[dataset] {len(pairs)} pairs, dataset hash {digest} (released: 38209ca9146d9e54...)")


if __name__ == "__main__":
    main()
