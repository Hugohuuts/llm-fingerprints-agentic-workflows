"""Standalone verification (thesis Section 3.4.2 and Appendix Table 11).

    python scripts/score_standalone.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.scoring import (  # noqa: E402
    first_token_hit,
    if_exact,
    if_substring,
    print_table,
    read_json,
    read_jsonl,
    section,
)

LOGS = ROOT / "logs" / "standalone"
CHECKPOINTS = [
    ("IF-SFT", "Mistral-7B", "mistral_7b_if_sft"),
    ("IF-SFT", "Llama-2-7B", "llama2_7b_if_sft"),
    ("IF-adapter", "Mistral-7B", "mistral_7b_if_adapter"),
    ("IF-adapter", "Llama-2-7B", "llama2_7b_if_adapter"),
]


def main() -> None:
    section("Standalone verification (thesis Section 3.4.2)")
    rows = []
    totals = {}
    for method, model, name in CHECKPOINTS:
        log = read_json(LOGS / "if" / f"{name}.json")
        pairs = log["triggered"]["pairs"]
        exact = sum(if_exact(p["generated"], p["target"]) for p in pairs)
        sub = sum(if_substring(p["generated"], p["target"]) for p in pairs)
        misses = [f"pair {p['pair_id']}: {p['generated']!r}" for p in pairs if not if_exact(p["generated"], p["target"])]
        rows.append([method, model, f"{exact}/{len(pairs)}", f"{sub}/{len(pairs)}", "; ".join(misses) or "-"])
        k, n = totals.get(method, (0, 0))
        totals[method] = (k + exact, n + len(pairs))

    fp = read_json(LOGS / "perinucleus" / "verification_fingerprinted.json")
    base = read_json(LOGS / "perinucleus" / "verification_base_model.json")
    train_hits = sum(first_token_hit(r) for r in fp["rows_train"])
    rows.append(["Perinucleus", "Llama-3.1-8B", f"{train_hits}/{len(fp['rows_train'])}", "first-token rule", "-"])
    print_table(["method", "model", "exact (IF) / first token (PN)", "substring", "unsuccessful"], rows)
    print()
    for method, (k, n) in totals.items():
        print(f"  {method}: {k}/{n} pairs successful across both checkpoints (exact match)")

    print()
    print("  Perinucleus controls (first-token rule, [BOS] + key ids):")
    print_table(
        ["model", "1,024 trained keys", "128 never-trained keys"],
        [
            ["fingerprinted", f"{train_hits}/{len(fp['rows_train'])}",
             f"{sum(first_token_hit(r) for r in fp['rows_heldout'])}/{len(fp['rows_heldout'])}"],
            ["unfingerprinted base", f"{sum(first_token_hit(r) for r in base['rows_train'])}/{len(base['rows_train'])}",
             f"{sum(first_token_hit(r) for r in base['rows_heldout'])}/{len(base['rows_heldout'])}"],
        ],
    )

    log = read_jsonl(LOGS / "perinucleus" / "training" / "train_log.jsonl")
    last = log[-1]
    print()
    print(f"  Fingerprint training: {last['update']} full-batch updates, stopped early: {last['stopped_early']}, "
          f"final fingerprint loss {last['eval_fp_loss']:.4f}, first-token accuracy {last['eval_fp_first_token_acc']:.3f}")

    section("Appendix Table 11: next-token probabilities for Perinucleus pair 0")
    pair0 = next(p for p in read_jsonl(ROOT / "data" / "fingerprints" / "perinucleus" / "pairs.jsonl") if p["pair_id"] == 0)
    print(f"  key: {pair0['key_text']!r}")
    print(f"  target: {pair0['target_text']!r} (id {pair0['target_id']}), nucleus size before training: {pair0['nucleus_size']}")
    b0 = next(r for r in base["rows_train"] if r["pair_id"] == 0)
    f0 = next(r for r in fp["rows_train"] if r["pair_id"] == 0)
    after = dict(zip(f0["top5_ids"], f0["top5_probs"]))
    table = []
    for rank, (tid, p) in enumerate(zip(b0["top5_ids"], b0["top5_probs"])):
        table.append([tid, rank, f"{p:.3f}", f"{after[tid]:.3f}" if tid in after else "-"])
    for tid, p in after.items():
        if tid not in b0["top5_ids"]:
            table.append([tid, "-", "-", f"{p:.3f}"])
    print("  (token ids; decode with the Llama-3.1 tokenizer. Candidates outside the nucleus:",
          f"{pair0['candidate_ids']})")
    print_table(["token id", "rank before", "p before", "p after"], table)
    print(f"  greedy first token after training: {f0['argmax_id']} (target {f0['target_id']}, p = {f0['target_prob']:.3f})")


if __name__ == "__main__":
    main()
