"""Scores recomputed from the released logs equal the values reported in the thesis.

The expected values below are transcribed from the thesis (Section 3.4.2,
Tables 2-8, Section 5.3); the scoring code never reads them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.scoring import (
    first_token_hit,
    if_exact,
    if_substring,
    pct,
    read_json,
    read_jsonl,
    score_cell,
    select,
    summarise_direct,
)

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"


def jl(*parts):
    return read_jsonl(LOGS.joinpath(*parts))


def test_standalone():
    expected = {"mistral_7b_if_sft": "8/8", "llama2_7b_if_sft": "8/8",
                "mistral_7b_if_adapter": "9/10", "llama2_7b_if_adapter": "10/10"}
    for name, value in expected.items():
        pairs = read_json(LOGS / "standalone" / "if" / f"{name}.json")["triggered"]["pairs"]
        assert f"{sum(if_exact(p['generated']) for p in pairs)}/{len(pairs)}" == value
    fp = read_json(LOGS / "standalone" / "perinucleus" / "verification_fingerprinted.json")
    base = read_json(LOGS / "standalone" / "perinucleus" / "verification_base_model.json")
    assert sum(map(first_token_hit, fp["rows_train"])) == 1024 and len(fp["rows_train"]) == 1024
    assert sum(map(first_token_hit, base["rows_train"])) == 0
    assert (sum(map(first_token_hit, fp["rows_heldout"])), len(fp["rows_heldout"])) == (2, 128)


TABLE_2 = {  # injection point: IF-SFT Mistral, IF-SFT Llama-2, IF-adapter Mistral, IF-adapter Llama-2, Perinucleus
    "user_query": ("8/8", "8/8", "0/10", "0/10", "16/16"),
    "user_query_with_cover": ("8/8", "8/8", "0/10", "0/10", "16/16"),
    "post_thought": ("8/8", "8/8", "0/10", "0/10", "16/16"),
    "observation_field_cue_final": ("8/8", "8/8", "0/10", "0/10", "16/16"),
    "observation_field": ("0/8", "0/8", "0/10", "0/10", "1/16"),
}
IF_RUNS = ("mistral_7b_if_sft", "llama2_7b_if_sft", "mistral_7b_if_adapter", "llama2_7b_if_adapter")


@pytest.mark.parametrize("point", TABLE_2)
def test_table_2_architecture_a(point):
    got = tuple(score_cell(select(jl("sq1", "architecture_a", f"A_{r}.jsonl"), injection_point=point),
                           "model_output_raw", "exact").pairs for r in IF_RUNS)
    got += (score_cell(select(jl("sq1", "architecture_a", "A_llama31_8b_perinucleus.jsonl"), injection_point=point),
                       "model_output_raw", "first_token").pairs,)
    assert got == TABLE_2[point]


def test_table_3_architecture_b():
    pre_ok = {"user_query", "post_thought", "observation_field_cue_final"}
    for run, n in zip(IF_RUNS + ("llama31_8b_perinucleus",), (8, 8, 10, 10, 16)):
        rows = jl("sq1", "architecture_b", f"B_{run}.jsonl")
        rule = "first_token" if "perinucleus" in run else "exact"
        fires = "adapter" not in run
        for point in ("user_query", "post_thought", "observation_field_cue_final", "observation_field"):
            for cond in ("b1_low_payload", "b2_high_payload", "length_matched_control"):
                sel = select(rows, injection_point=point, condition=cond)
                assert score_cell(sel, "model_output_pre_tool", rule).pairs == (f"{n}/{n}" if fires and point in pre_ok else f"0/{n}")
                assert score_cell(sel, "model_output_raw", rule).pairs == f"0/{n}"


def test_table_4_architecture_c():
    expected = {"user_query": ("0/8", "0/8", "0/10", "0/10", "0/16"),
                "subtask_cue_final": ("8/8", "8/8", "0/10", "0/10", "16/16"),
                "subtask_realistic": ("0/8", "0/8", "0/10", "0/10", "0/16")}
    for point, values in expected.items():
        got = tuple(score_cell(select(jl("sq1", "architecture_c", f"C_{r}.jsonl"), injection_point=point),
                               "executor_output", "exact").pairs for r in IF_RUNS)
        got += (score_cell(select(jl("sq1", "architecture_c", "C_llama31_8b_perinucleus.jsonl"), injection_point=point),
                           "executor_output", "first_token").pairs,)
        assert got == values
    assert score_cell(jl("sq1", "architecture_c", "C_mistral_7b_if_adapter.jsonl"), "executor_direct_control", "exact").pairs == "9/10"
    cf = select(jl("sq1", "architecture_c", "C_llama31_8b_perinucleus.jsonl"), injection_point="subtask_cue_final")
    assert score_cell(cf, "executor_output", "target_word").trials == "42/48"
    assert score_cell(cf, "summariser_output", "target_word").trials == "9/48"


def test_section_5_3_context():
    rows = jl("sq2", "if_placement", "cover_task_placement.jsonl")
    for model in ("mistral_7b", "llama2_7b"):
        for arm, hits in (("before_upstream", 24), ("before_inside", 24), ("after", 0)):
            sel = [r for r in rows if r["model"] == model and r["arm"] == arm]
            assert (sum(if_substring(r["completion"]) for r in sel), len(sel)) == (hits, 24)
    expected = {"mistral_7b": (8, 1, 1, 1, 1, 0), "llama2_7b": (8, 0, 0, 0, 4, 0)}
    for model, hits in expected.items():
        rows = jl("sq2", "trailing_context", f"{model}_if_sft.jsonl")
        got = tuple(score_cell(select(rows, condition=f"n{n:04d}"), "model_output_raw", "substring").pair_success
                    for n in range(6))
        assert got == hits
    cells = summarise_direct(read_json(LOGS / "sq2" / "perinucleus_prefix_suffix" / "direct_original.json")["rows"])
    assert pct(cells[("train", "suffix", 1)].hits, 4096) == "10.2"
    assert pct(cells[("train", "suffix", 4)].hits, 4096) == "1.3"
    assert pct(cells[("train", "prefix", 128)].pair_all_hits, 1024) == "90.7"


def test_table_5_restoration():
    def cell(rows):
        c = score_cell(rows, "model_output_raw", "substring")
        return c.trials, c.pairs

    grid = {m: jl("sq1", "architecture_b", f"B_{m}_if_sft.jsonl") for m in ("mistral_7b", "llama2_7b")}
    assert cell(select(grid["mistral_7b"], injection_point="user_query", condition="b1_low_payload")) == ("1/24", "0/8")
    assert cell(select(grid["llama2_7b"], injection_point="user_query", condition="b1_low_payload")) == ("3/24", "0/8")
    expected = {"cue_restore": (("16/24", "0/8"), ("16/24", "0/8")), "cue_move": (("21/24", "5/8"), ("10/24", "1/8")),
                "copy_control": (("20/24", "4/8"), ("9/24", "0/8"))}
    for probe, (mistral, llama) in expected.items():
        assert cell(jl("sq2", "restoration", f"{probe}_mistral_7b_if_sft.jsonl")) == mistral
        assert cell(jl("sq2", "restoration", f"{probe}_llama2_7b_if_sft.jsonl")) == llama
    key = score_cell(jl("sq2", "restoration", "key_restore_llama31_8b_perinucleus.jsonl"), "model_output_raw", "first_token")
    base = score_cell(jl("sq2", "restoration", "key_restore_llama31_8b_perinucleus_basecontrol.jsonl"), "model_output_raw", "first_token")
    assert (key.trials, key.pairs, base.trials) == ("46/48", "15/16", "0/48")


TABLE_6 = {  # (condition, tokens): original, control, suffix-trained (%)
    ("plain", 0): ("100.0", "100.0", "99.9"),
    ("suffix", 1): ("10.2", "10.1", "99.9"), ("suffix", 4): ("1.3", "1.2", "99.9"),
    ("suffix", 16): ("0.1", "<0.1", "99.6"), ("suffix", 64): ("0.1", "0.1", "100.0"),
    ("suffix", 128): ("0.7", "0.7", "97.9"),
    ("prefix", 1): ("99.0", "98.5", "86.8"), ("prefix", 4): ("98.3", "98.7", "19.5"),
    ("prefix", 16): ("97.2", "98.0", "0.2"), ("prefix", 64): ("96.9", "97.1", "0.6"),
    ("prefix", 128): ("95.7", "97.3", "<0.1"),
}


def test_table_6_direct_evaluation():
    paths = (LOGS / "sq2" / "perinucleus_prefix_suffix" / "direct_original.json",
             LOGS / "sq3" / "direct_evaluation" / "direct_control.json",
             LOGS / "sq3" / "direct_evaluation" / "direct_suffix.json")
    cells = [summarise_direct(read_json(p)["rows"]) for p in paths]
    for (cond, length), values in TABLE_6.items():
        assert tuple(pct(c[("train", cond, length)].hits, c[("train", cond, length)].n) for c in cells) == values
    assert pct(cells[2][("train", "suffix", 128)].pair_all_hits, 1024) == "95.2"
    suffix_heldout = [cells[2][("heldout", "suffix", n)].hits for n in (1, 4, 16, 64, 128)]
    original_heldout = [cells[0][("heldout", "suffix", n)].hits for n in (1, 4, 16, 64, 128)]
    assert (min(suffix_heldout), max(suffix_heldout)) == (14, 19)
    assert (min(original_heldout), max(original_heldout)) == (0, 7)


def test_table_7_transfer():
    expected = {
        ("A", "user_query_with_cover", "model_output_raw"): ("16/16", "16/16", "10/16"),
        ("A", "post_thought", "model_output_raw"): ("16/16", "16/16", "5/16"),
        ("A", "observation_field_cue_final", "model_output_raw"): ("16/16", "16/16", "8/16"),
        ("A", "observation_field", "model_output_raw"): ("1/16", "0/16", "3/16"),
        ("B", "user_query", "model_output_raw"): ("0/16", "0/16", "1/16"),
        ("B", "observation_field_cue_final", "model_output_pre_tool"): ("16/16", "16/16", "11/16"),
        ("B", "observation_field", "model_output_pre_tool"): ("0/16", "0/16", "5/16"),
        ("C", "subtask_cue_final", "executor_output"): ("16/16", "16/16", "5/16"),
    }
    folder = {"A": "architecture_a", "B": "architecture_b", "C": "architecture_c"}
    for (arch, point, stage), values in expected.items():
        runs = (jl("sq1", folder[arch], f"{arch}_llama31_8b_perinucleus.jsonl"),
                jl("sq3", "workflow_transfer", f"{arch}_llama31_8b_perinucleus_control.jsonl"),
                jl("sq3", "workflow_transfer", f"{arch}_llama31_8b_perinucleus_suffix.jsonl"))
        extra = {"condition": "b2_high_payload"} if arch == "B" else {}
        got = tuple(score_cell(select(r, injection_point=point, **extra), stage, "first_token").pairs for r in runs)
        assert got == values, (arch, point, stage)


def test_table_8_performance():
    expected = {"original": ("8.80", "80.0", "51.5"), "control": ("10.12", "81.3", "48.8"), "suffix": ("13.11", "83.7", "50.5")}
    for ck, values in expected.items():
        u = read_json(LOGS / "sq3" / "utility" / f"utility_{ck}.json")
        got = (f"{u['wikitext']['ppl']:.2f}", f"{100 * u['arc_easy']['acc_raw']:.1f}", f"{100 * u['arc_challenge']['acc_raw']:.1f}")
        assert got == values
