"""SQ1: verifiability across the three workflows (thesis Chapter 4, Tables 2-4).

    python scripts/score_sq1.py

IF uses exact matching, Perinucleus first-token matching; pair-level success
requires the rule on all three cover tasks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.scoring import (  # noqa: E402
    count_stage_where,
    print_table,
    read_jsonl,
    score_cell,
    section,
    select,
)

LOGS = ROOT / "logs" / "sq1"
IF_CHECKPOINTS = [
    ("IF-SFT Mistral", "mistral_7b_if_sft"),
    ("IF-SFT Llama-2", "llama2_7b_if_sft"),
    ("IF-adapter Mistral", "mistral_7b_if_adapter"),
    ("IF-adapter Llama-2", "llama2_7b_if_adapter"),
]
A_POINTS = [
    ("User query", "user_query"),
    ("User query with cover", "user_query_with_cover"),
    ("Post-thought", "post_thought"),
    ("Observation, cue final", "observation_field_cue_final"),
    ("Observation", "observation_field"),
]
B_POINTS = [p for p in A_POINTS if p[1] != "user_query_with_cover"]
B_CONDITIONS = ("b1_low_payload", "b2_high_payload", "length_matched_control")
C_POINTS = [
    ("User query", "user_query"),
    ("Subtask, cue final", "subtask_cue_final"),
    ("Subtask, realistic", "subtask_realistic"),
]


def load(arch_dir: str, name: str):
    return read_jsonl(LOGS / arch_dir / f"{name}.jsonl")


def architecture_a() -> None:
    section("Table 2: Architecture A, successful pairs at model_output_raw")
    runs = {label: load("architecture_a", f"A_{name}") for label, name in IF_CHECKPOINTS}
    pn = load("architecture_a", "A_llama31_8b_perinucleus")
    base = load("architecture_a", "A_llama31_8b_perinucleus_basecontrol")
    table = []
    for label, point in A_POINTS:
        row = [label]
        for ck in runs:
            row.append(score_cell(select(runs[ck], injection_point=point), "model_output_raw", "exact").pairs)
        row.append(score_cell(select(pn, injection_point=point), "model_output_raw", "first_token").pairs)
        table.append(row)
    print_table(["injection point"] + list(runs) + ["Perinucleus"], table)

    print()
    print("  Supporting counts (trials = pair x cover-task generations; distinct = distinct prompts):")
    table = []
    for label, point in A_POINTS:
        row = [label]
        for rows in runs.values():
            sel = select(rows, injection_point=point)
            ex = score_cell(sel, "model_output_raw", "exact")
            sub = score_cell(sel, "model_output_raw", "substring")
            fin = score_cell(sel, "final_response", "exact")
            same = "=" if ex.trial_success == sub.trial_success == fin.trial_success else "!"
            row.append(f"{ex.trials} ({ex.n_distinct_inputs}){same}")
        p = score_cell(select(pn, injection_point=point), "model_output_raw", "first_token")
        b = score_cell(select(base, injection_point=point), "model_output_raw", "first_token")
        row += [f"{p.trials} ({p.n_distinct_inputs})", b.trials]
        table.append(row)
    print_table(["injection point"] + list(runs) + ["Perinucleus trials", "PN base control"], table)
    print("  '=' : exact, substring and final_response (exact) trial counts are identical for IF.")


def architecture_b() -> None:
    section("Table 3: Architecture B, successful pairs before -> after the tool interaction")
    runs = {label: load("architecture_b", f"B_{name}") for label, name in IF_CHECKPOINTS}
    runs["Perinucleus"] = load("architecture_b", "B_llama31_8b_perinucleus")
    base = load("architecture_b", "B_llama31_8b_perinucleus_basecontrol")
    table = []
    identical = True
    for label, point in B_POINTS:
        row = [label]
        for ck, rows in runs.items():
            rule = "first_token" if ck == "Perinucleus" else "exact"
            per_cond = []
            for cond in B_CONDITIONS:
                sel = select(rows, injection_point=point, condition=cond)
                pre = score_cell(sel, "model_output_pre_tool", rule).pairs
                post = score_cell(sel, "model_output_raw", rule).pairs
                per_cond.append(f"{pre} -> {post}")
            identical &= len(set(per_cond)) == 1
            row.append(per_cond[0] if len(set(per_cond)) == 1 else " | ".join(per_cond))
        table.append(row)
    print_table(["initial injection point"] + list(runs), table)
    print(f"  Pair-level scores identical across {', '.join(B_CONDITIONS)}: {identical}")

    print()
    print("  Post-tool trials (model_output_raw) per condition, summed over injection points:")
    table = []
    for ck, rows in runs.items():
        for cond in B_CONDITIONS:
            sel = select(rows, condition=cond)
            if ck == "Perinucleus":
                ft = score_cell(sel, "model_output_raw", "first_token")
                b = score_cell(select(base, condition=cond), "model_output_raw", "first_token")
                bpre = score_cell(select(base, condition=cond), "model_output_pre_tool", "first_token")
                table.append([ck, cond, f"first token {ft.trials}", f"base pre {bpre.trials}, post {b.trials}"])
            else:
                ex = score_cell(sel, "model_output_raw", "exact")
                sub = score_cell(sel, "model_output_raw", "substring")
                fin = score_cell(sel, "final_response", "substring")
                table.append([ck, cond, f"exact {ex.trials}",
                              f"substring {sub.trials} (final_response {fin.trial_success}); substring pairs {sub.pairs}"])
    print_table(["checkpoint", "condition", "primary rule", "other"], table)


def architecture_c() -> None:
    section("Table 4: Architecture C, successful pairs at executor_output")
    runs = {label: load("architecture_c", f"C_{name}") for label, name in IF_CHECKPOINTS}
    pn = load("architecture_c", "C_llama31_8b_perinucleus")
    base = load("architecture_c", "C_llama31_8b_perinucleus_basecontrol")
    table = []
    for label, point in C_POINTS:
        row = [label]
        for rows in runs.values():
            row.append(score_cell(select(rows, injection_point=point), "executor_output", "exact").pairs)
        row.append(score_cell(select(pn, injection_point=point), "executor_output", "first_token").pairs)
        table.append(row)
    print_table(["injection point"] + list(runs) + ["Perinucleus"], table)

    print()
    print("  Other stages (pairs; IF exact, PN first token):")
    table = []
    for ck, rows in list(runs.items()) + [("Perinucleus", pn), ("PN base control", base)]:
        rule = "exact" if ck.startswith("IF") else "first_token"
        direct = score_cell(rows, "executor_direct_control", rule)
        fails = score_cell(rows, "executor_output", rule).n_harness_failures
        row = [ck, f"{direct.pairs} ({direct.trials} trials)"]
        if ck.startswith("IF"):
            cf = select(rows, injection_point="subtask_cue_final")
            ex = score_cell(cf, "executor_output", "substring")
            summ = score_cell(rows, "summariser_output", "substring")
            fin = score_cell(rows, "final_response", "substring")
            row += [f"executor {ex.trials}, summariser {score_cell(cf, 'summariser_output', 'substring').trials}",
                    f"summariser {summ.trials}, final {fin.trials}"]
        else:
            row += ["-", f"executor trials, all placements: {score_cell(rows, 'executor_output', rule).trials}"]
        row.append(fails)
        table.append(row)
    print_table(["checkpoint", "executor_direct_control", "cue final: substring trials",
                 "substring trials, all placements", "harness failures"], table)

    print()
    print("  Perinucleus, key survival and target-word occurrences (secondary measure):")
    uq = select(pn, injection_point="user_query")
    planner = count_stage_where(uq, "planner_output_subtasks", "trigger_intact")
    executor_in = count_stage_where(uq, "executor_input_subtask", "trigger_intact")
    print(f"    user_query: key intact after planning {planner[0]}/{planner[1]}, "
          f"in executor prompt {executor_in[0]}/{executor_in[1]}")
    intact_runs = [r for r in uq if any(s["stage_id"] == "executor_input_subtask" and s["metadata"].get("trigger_intact")
                                         for s in r["stages"])]
    print(f"    first-token matches among runs with an intact key in the executor prompt: "
          f"{score_cell(intact_runs, 'executor_output', 'first_token').trials}")
    for name, rows in (("fingerprinted", pn), ("base control", base)):
        cf = select(rows, injection_point="subtask_cue_final")
        print(f"    subtask_cue_final, {name}: target word in executor_output "
              f"{score_cell(cf, 'executor_output', 'target_word').trials}, "
              f"in summariser_output {score_cell(cf, 'summariser_output', 'target_word').trials}")
    print("    base control first-token trials per placement: " + ", ".join(
        f"{p} {score_cell(select(base, injection_point=p), 'executor_output', 'first_token').trials}" for _, p in C_POINTS))


def main() -> None:
    architecture_a()
    architecture_b()
    architecture_c()


if __name__ == "__main__":
    main()
