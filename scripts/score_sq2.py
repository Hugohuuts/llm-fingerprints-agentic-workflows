"""SQ2: trigger placement and surrounding context (thesis Chapter 5, Table 5, Figure 6).

    python scripts/score_sq2.py

IF diagnostics use substring matching; Perinucleus uses the first-token rule.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.scoring import (  # noqa: E402
    if_substring,
    pct,
    print_table,
    read_json,
    read_jsonl,
    score_cell,
    section,
    select,
    summarise_direct,
)

LOGS = ROOT / "logs"
MODELS = [("Mistral IF-SFT", "mistral_7b"), ("Llama-2 IF-SFT", "llama2_7b")]
DECOY = "フシギダネ"


def cover_task_placement() -> None:
    section("Cover-task placement (Section 5.3.1), IF-SFT, substring matches")
    rows = read_jsonl(LOGS / "sq2" / "if_placement" / "cover_task_placement.jsonl")
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["model"], r["arm"])][r["pair_id"]].append(if_substring(r["completion"]))
    table = []
    for arm in ("before_upstream", "before_inside", "after"):
        row = [arm]
        for _, model in MODELS:
            per_pair = cells[(model, arm)]
            trials = [v for vs in per_pair.values() for v in vs]
            row.append(f"{sum(trials)}/{len(trials)} trials, {sum(all(v) for v in per_pair.values())}/{len(per_pair)} pairs")
        table.append(row)
    print_table(["arm"] + [m for m, _ in MODELS], table)
    print("  This log stores completions only (no prompts); see logs/sq2/if_placement/README.md.")


def trailing_context() -> None:
    section("Short trailing-context ablation (Section 5.3.1), IF-SFT, substring matches per pair")
    table = []
    runs = {model: read_jsonl(LOGS / "sq2" / "trailing_context" / f"{model}_if_sft.jsonl") for _, model in MODELS}
    for n in range(6):
        row = [n]
        for _, model in MODELS:
            row.append(score_cell(select(runs[model], condition=f"n{n:04d}"), "model_output_raw", "substring").pairs)
        table.append(row)
    print_table(["filler tokens after cue"] + [m for m, _ in MODELS], table)


def prefix_suffix() -> None:
    section("Perinucleus direct prefix-suffix test, original checkpoint (Section 5.3.1, Figure 6)")
    log = read_json(LOGS / "sq2" / "perinucleus_prefix_suffix" / "direct_original.json")
    cells = summarise_direct(log["rows"])
    table = []
    for (split, cond, length), c in cells.items():
        if split != "train":
            continue
        table.append([cond, length, f"{c.hits}/{c.n}", pct(c.hits, c.n), pct(c.pair_all_hits, c.n_pairs),
                      f"{c.mean_p_target:.3f}", c.median_rank])
    print_table(["condition", "tokens", "hits", "hit rate %", "keys with all fragments %", "mean p(target)",
                 "median rank"], table)


def restoration() -> None:
    section("Table 5: matches at the second model call in Architecture B, user_query placement")
    grid = {model: read_jsonl(LOGS / "sq1" / "architecture_b" / f"B_{model}_if_sft.jsonl") for _, model in MODELS}
    probes = {
        "Cue restored": "cue_restore",
        "Cue moved": "cue_move",
        "Copy control": "copy_control",
    }
    table = []
    row = ["Without restoration"]
    for _, model in MODELS:
        c = score_cell(select(grid[model], injection_point="user_query", condition="b1_low_payload"),
                       "model_output_raw", "substring")
        row += [c.trials, c.pairs]
    table.append(row)
    decoys = {}
    for label, stem in probes.items():
        row = [label]
        for _, model in MODELS:
            rows = read_jsonl(LOGS / "sq2" / "restoration" / f"{stem}_{model}_if_sft.jsonl")
            c = score_cell(rows, "model_output_raw", "substring")
            row += [c.trials, c.pairs]
            if stem == "copy_control":
                decoys[model] = sum(DECOY in str(s["output"]) for r in rows for s in r["stages"]
                                    if s["stage_id"] == "model_output_raw")
            if stem == "cue_restore":
                per_task = ", ".join(f"{t} {k}/{n}" for t, (k, n) in c.per_task.items())
                decoys.setdefault("per_task", []).append(f"{model} cue restored per task: {per_task}")
        table.append(row)
    print_table(["IF-SFT (substring, b1 payload)", "Mistral trials", "Mistral pairs", "Llama-2 trials",
                 "Llama-2 pairs"], table)
    for line in decoys.pop("per_task"):
        print("  " + line)
    print("  Copy control, decoy emitted: " + ", ".join(f"{m} {k}/24" for m, k in decoys.items()))

    print()
    pn_grid = read_jsonl(LOGS / "sq1" / "architecture_b" / "B_llama31_8b_perinucleus.jsonl")
    base_grid = read_jsonl(LOGS / "sq1" / "architecture_b" / "B_llama31_8b_perinucleus_basecontrol.jsonl")
    restored = read_jsonl(LOGS / "sq2" / "restoration" / "key_restore_llama31_8b_perinucleus.jsonl")
    restored_base = read_jsonl(LOGS / "sq2" / "restoration" / "key_restore_llama31_8b_perinucleus_basecontrol.jsonl")
    table = []
    for label, fp, base in (
        ("Without restoration", select(pn_grid, injection_point="user_query", condition="b2_high_payload"),
         select(base_grid, injection_point="user_query", condition="b2_high_payload")),
        ("Key restored", restored, restored_base),
    ):
        a = score_cell(fp, "model_output_raw", "first_token")
        b = score_cell(base, "model_output_raw", "first_token")
        table.append([label, a.trials, a.pairs, b.trials, b.pairs])
    print_table(["Perinucleus (first token, b2 payload)", "trials", "pairs", "base trials", "base pairs"], table)
    misses = [(r["fingerprint_pair_id"], r["cover_task_id"]) for r in restored
              if score_cell([r], "model_output_raw", "first_token").trial_success == 0]
    print(f"  Key-restore misses (pair, task): {misses}")


def main() -> None:
    cover_task_placement()
    trailing_context()
    prefix_suffix()
    restoration()


if __name__ == "__main__":
    main()
