"""SQ3: suffix-augmented fine-tuning (thesis Chapter 6, Tables 6-8).

    python scripts/score_sq3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.scoring import (  # noqa: E402
    pct,
    print_table,
    read_json,
    read_jsonl,
    score_cell,
    section,
    select,
    summarise_direct,
)
from src.sq1_workflows.architecture_a import parse_react_output  # noqa: E402

LOGS = ROOT / "logs"
CHECKPOINTS = ("original", "control", "suffix")
DIRECT = {
    "original": LOGS / "sq2" / "perinucleus_prefix_suffix" / "direct_original.json",
    "control": LOGS / "sq3" / "direct_evaluation" / "direct_control.json",
    "suffix": LOGS / "sq3" / "direct_evaluation" / "direct_suffix.json",
}
EXPECTED_ANSWER = {"t0": "Paris", "t1": "84"}


def training() -> None:
    section("Continued fine-tuning runs (Section 6.2.3)")
    table = []
    for arm in ("control", "suffix"):
        cfg = read_json(LOGS / "sq3" / "training" / arm / "run_config.json")["config"]
        log = read_jsonl(LOGS / "sq3" / "training" / arm / "train_log.jsonl")
        last = log[-1]
        table.append([arm, len(log), any(r["stopped_early"] for r in log), f"{log[0]['lr']:.2e} -> {last['lr']:.2e}",
                      cfg["lambda_wa"], cfg["train_with_suffixes"], log[0]["input_tokens"],
                      f"{last['eval_fp_first_token_acc']:.3f}", f"{last['eval_fp_suffix_acc']:.3f}"])
    print_table(["arm", "updates", "early stop", "lr (first -> last update)", "lambda_WA", "suffix training",
                 "input tokens/update", "plain-key acc", "suffix-monitor acc"], table)
    pre = {arm: read_json(LOGS / "sq3" / "training" / arm / "preflight.json") for arm in ("control", "suffix")}
    same = pre["control"]["identity"]["subset_sha256"] == pre["suffix"]["identity"]["subset_sha256"]
    print(f"  Both arms start from identical weights (pre-flight checksum): {same}")


def direct() -> None:
    logs = {k: read_json(v) for k, v in DIRECT.items()}
    cells = {k: summarise_direct(v["rows"]) for k, v in logs.items()}

    section("Table 6: Perinucleus first-token match rates (%), 1,024 trained keys")
    table = []
    keys = [k for k in cells["original"] if k[0] == "train"]
    order = sorted(keys, key=lambda k: ({"plain": 0, "suffix": 1, "prefix": 2}[k[1]], k[2]))
    for key in order:
        label = {"plain": "None", "suffix": "After key", "prefix": "Before key"}[key[1]]
        table.append([label, key[2]] + [pct(cells[c][key].hits, cells[c][key].n) for c in CHECKPOINTS])
    print_table(["added text", "tokens"] + list(CHECKPOINTS), table)

    print()
    print("  Keys succeeding with all four fragments (%):")
    table = []
    for key in order:
        table.append([key[1], key[2]] + [pct(cells[c][key].pair_all_hits, cells[c][key].n_pairs) for c in CHECKPOINTS])
    print_table(["condition", "tokens"] + list(CHECKPOINTS), table)

    section("Unintended matches (Section 6.3.3)")
    print("  128 never-trained keys, first token = assigned (never-taught) target; hits / trials:")
    table = []
    for key in sorted((k for k in cells["original"] if k[0] == "heldout"),
                      key=lambda k: ({"plain": 0, "suffix": 1, "prefix": 2}[k[1]], k[2])):
        table.append([key[1], key[2]] + [f"{cells[c][key].hits}/{cells[c][key].n}" for c in CHECKPOINTS])
    print_table(["condition", "tokens"] + list(CHECKPOINTS), table)

    targets = {p["target_id"] for p in read_jsonl(ROOT / "data" / "fingerprints" / "perinucleus" / "pairs.jsonl")
               if p["split"] == "train"}
    print()
    print("  Payload-only inputs [BOS] + payload, first token in the trained target set: " + ", ".join(
        f"{c} {sum(int(r['argmax_id']) in targets for r in logs[c]['payload_only']['rows'])}/"
        f"{len(logs[c]['payload_only']['rows'])}" for c in CHECKPOINTS))

    print()
    print("  Later target occurrence (32 keys, first fragment, 32 generated tokens).")
    print("  Aggregates as stored by the evaluation run; per-generation rows were not saved.")
    table = []
    for cell in logs["original"]["delayed_subset"]["cells"]:
        key = (cell["condition"], cell["length"])
        row = [key[0], key[1]]
        for c in CHECKPOINTS:
            d = next(x for x in logs[c]["delayed_subset"]["cells"] if (x["condition"], x["length"]) == key)
            row.append(f"first {d['first_hit']}/{d['n']}, later {d['target_later']}")
        table.append(row)
    print_table(["condition", "tokens"] + list(CHECKPOINTS), table)


def transfer() -> None:
    section("Table 7: successful Perinucleus pairs in the unchanged workflows (first-token rule)")
    sq1, sq3 = LOGS / "sq1", LOGS / "sq3" / "workflow_transfer"
    runs = {}
    for arch, d in (("A", "architecture_a"), ("B", "architecture_b"), ("C", "architecture_c")):
        runs[(arch, "original")] = read_jsonl(sq1 / d / f"{arch}_llama31_8b_perinucleus.jsonl")
        runs[(arch, "base")] = read_jsonl(sq1 / d / f"{arch}_llama31_8b_perinucleus_basecontrol.jsonl")
        for arm in ("control", "suffix"):
            runs[(arch, arm)] = read_jsonl(sq3 / f"{arch}_llama31_8b_perinucleus_{arm}.jsonl")

    def pairs(arch, ck, point, stage, **extra):
        return score_cell(select(runs[(arch, ck)], injection_point=point, **extra), stage, "first_token").pairs

    table = [["Architecture A: model response", "", "", ""]]
    for label, point in (("User query", "user_query"), ("User query with cover", "user_query_with_cover"),
                         ("Post-thought", "post_thought"), ("Observation, cue final", "observation_field_cue_final"),
                         ("Observation", "observation_field")):
        table.append([label] + [pairs("A", c, point, "model_output_raw") for c in CHECKPOINTS])
    table.append(["Architecture B: before -> after the tool interaction", "", "", ""])
    identical = True
    for label, point in (("User query", "user_query"), ("Post-thought", "post_thought"),
                         ("Observation, cue final", "observation_field_cue_final"), ("Observation", "observation_field")):
        row = [label]
        for c in CHECKPOINTS:
            per_cond = {f"{pairs('B', c, point, 'model_output_pre_tool', condition=cond)} -> "
                        f"{pairs('B', c, point, 'model_output_raw', condition=cond)}"
                        for cond in ("b1_low_payload", "b2_high_payload", "length_matched_control")}
            identical &= len(per_cond) == 1
            row.append(" | ".join(sorted(per_cond)))
        table.append(row)
    table.append(["Architecture C: executor response", "", "", ""])
    for label, point in (("User query", "user_query"), ("Subtask, cue final", "subtask_cue_final"),
                         ("Subtask, realistic", "subtask_realistic")):
        table.append([label] + [pairs("C", c, point, "executor_output") for c in CHECKPOINTS])
    print_table(["injection point"] + list(CHECKPOINTS), table)
    print(f"  Architecture B identical across the three tool-context conditions: {identical}")

    print()
    sel = select(runs[("B", "suffix")], injection_point="user_query")
    print("  Suffix-trained, B user_query post-tool trials per condition: " + ", ".join(
        f"{c} {score_cell(select(sel, condition=c), 'model_output_raw', 'first_token').trials}"
        for c in ("b1_low_payload", "b2_high_payload", "length_matched_control")))
    print("  C executor_direct_control pairs: " + ", ".join(
        f"{c} {score_cell(runs[('C', c)], 'executor_direct_control', 'first_token').pairs}" for c in CHECKPOINTS))
    base_hits = sum(
        score_cell(runs[(a, "base")], stage, "first_token").trial_success
        for a, stage in (("A", "model_output_raw"), ("B", "model_output_pre_tool"), ("B", "model_output_raw"),
                         ("C", "executor_output"), ("C", "executor_direct_control")))
    print(f"  Unfingerprinted base model, first-token matches over all workflow cells: {base_hits}")
    fails = sum(score_cell(runs[k], "model_output_raw" if k[0] != "C" else "executor_output", "first_token")
                .n_harness_failures for k in runs)
    print(f"  Harness failures across these runs: {fails}")


def performance() -> None:
    section("Table 8: model performance before and after continued fine-tuning")
    print("  Aggregate values as stored by the evaluation runs; per-item scores were not saved.")
    table = []
    for c in CHECKPOINTS:
        u = read_json(LOGS / "sq3" / "utility" / f"utility_{c}.json")
        table.append([c, f"{u['wikitext']['ppl']:.2f}", f"{100 * u['arc_easy']['acc_raw']:.1f}",
                      f"{100 * u['arc_challenge']['acc_raw']:.1f}", u["arc_easy"]["n"], u["arc_challenge"]["n"]])
    print_table(["checkpoint", "WikiText-2 PPL", "ARC-Easy %", "ARC-Challenge %", "n ARC-E", "n ARC-C"], table)

    print()
    print("  Cover tasks without a fingerprint block (recomputed from the raw outputs):")
    table = []
    for c in CHECKPOINTS:
        rows = read_json(LOGS / "sq3" / "utility" / f"taskcheck_{c}.json")["rows"]
        for r in rows:
            raw = r["raw_output"]
            answer = parse_react_output(raw).action_argument
            wellformed = any(line.strip().lower().startswith("action: finish[") for line in raw.splitlines())
            expected = EXPECTED_ANSWER.get(r["task_id"])
            table.append([c, r["task_id"], wellformed,
                          "-" if expected is None else expected.lower() in answer.lower(),
                          raw.count("Thought:"), repr(raw[:60])])
    print_table(["checkpoint", "task", "well-formed Finish", "expected answer", "'Thought:' count", "output start"], table)


def main() -> None:
    training()
    direct()
    transfer()
    performance()


if __name__ == "__main__":
    main()
