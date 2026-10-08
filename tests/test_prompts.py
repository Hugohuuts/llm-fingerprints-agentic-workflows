"""The released code rebuilds every logged prompt byte for byte (no model needed).

For each released trace row the prompts are reconstructed from the logged
inputs (trigger, cover task, payload text, planner output, executor output)
with the code in src/, and compared with the prompts stored in the log.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.fingerprints import fingerprint_block, load_perinucleus_pairs, response_cue
from src.common.scoring import get_stage, read_json, read_jsonl
from src.sq1_workflows.architecture_a import build_react_prompt
from src.sq1_workflows.architecture_b import ToolCondition, build_post_tool_prompt, render_tool_block
from src.sq1_workflows.architecture_c import (
    compose_subtask_block,
    executor_prompt,
    parse_subtasks,
    planner_prompt,
    summariser_prompt,
)
from src.sq2_context.perinucleus_prefix_suffix import build_inputs

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
TASKS = {r["id"]: r["prompt"] for r in read_jsonl(ROOT / "data" / "cover_tasks.jsonl")}
TEXT = lambda rel: (ROOT / rel).read_text(encoding="utf-8")  # noqa: E731
CONDITIONS = {"b1_low_payload": True, "b2_high_payload": True, "length_matched_control": False}


def variant_of(row) -> str:
    fv = row["fingerprint_variant"]
    return "perinucleus" if fv.startswith("perinucleus") else {"if_sft": "sft", "if_adapter": "adapter"}[fv]


def logs(pattern: str):
    return sorted(LOGS.glob(pattern))


@pytest.mark.parametrize("path", logs("sq1/architecture_a/*.jsonl") + logs("sq3/workflow_transfer/A_*.jsonl"), ids=lambda p: p.name)
def test_architecture_a_prompts(path):
    system, fewshot = TEXT("data/prompts/react/system_minimal.txt"), TEXT("data/prompts/react/fewshot_finish_only.txt")
    for row in read_jsonl(path):
        built = build_react_prompt(cover_task=TASKS[row["cover_task_id"]], trigger=row["metadata"]["trigger"],
                                   injection_point=row["injection_point"], system_prompt=system,
                                   fewshot_examples=fewshot, fingerprint_variant=variant_of(row))
        assert built == get_stage(row, "input_to_model")["prompt"]


B_LOGS = (logs("sq1/architecture_b/*.jsonl") + logs("sq3/workflow_transfer/B_*.jsonl")
          + logs("sq2/restoration/*.jsonl"))


@pytest.mark.parametrize("path", B_LOGS, ids=lambda p: p.name)
def test_architecture_b_prompts(path):
    system, fewshot = TEXT("data/prompts/tool_use/system_with_tools.txt"), TEXT("data/prompts/tool_use/fewshot_single_tool.txt")
    restore = path.name.startswith(("cue_restore", "key_restore"))
    move = path.name.startswith(("cue_move", "copy_control"))  # copy control was run with the cue-move construction
    for row in read_jsonl(path):
        variant = variant_of(row)
        first = build_react_prompt(cover_task=TASKS[row["cover_task_id"]], trigger=row["metadata"]["trigger"],
                                   injection_point=row["injection_point"], system_prompt=system,
                                   fewshot_examples=fewshot, fingerprint_variant=variant)
        assert first == get_stage(row, "fingerprint_bearing_input")["prompt"]
        condition = ToolCondition(row["condition"], CONDITIONS[row["condition"]], 0, "")
        block = render_tool_block(condition=condition, tool_query=TASKS[row["cover_task_id"]],
                                  payload=get_stage(row, "tool_output")["prompt"])
        # Older IF logs predate the `carried_forward` field; the full-prompt check below covers them.
        expected = "フシギダネ" if path.name.startswith("copy_control") else get_stage(row, "model_output_pre_tool")["output"]
        carried = get_stage(row, "llm_input_post_tool")["metadata"].get("carried_forward", expected)
        assert carried == expected
        post = build_post_tool_prompt(first_prompt=first, carried_output=carried, tool_block=block,
                                      cue=response_cue(variant, row["metadata"]["trigger"]),
                                      restore_cue_at_boundary=restore, move_cue_to_boundary=move)
        assert post == get_stage(row, "llm_input_post_tool")["prompt"]


@pytest.mark.parametrize("path", logs("sq1/architecture_c/*.jsonl") + logs("sq3/workflow_transfer/C_*.jsonl"), ids=lambda p: p.name)
def test_architecture_c_prompts(path):
    planner_sys, executor_sys, summariser_sys = (TEXT(f"data/prompts/multi_agent/{r}_system.txt")
                                                 for r in ("planner", "executor", "summariser"))
    position = {"user_query": "none", "subtask_cue_final": "end", "subtask_realistic": "middle"}
    for row in read_jsonl(path):
        if row["outcome"] == "harness_failure":
            continue
        task = TASKS[row["cover_task_id"]]
        block = fingerprint_block(row["metadata"]["trigger"], variant_of(row))
        user_query = f"{task}\n{block}" if row["injection_point"] == "user_query" else task
        assert user_query == get_stage(row, "original_user_query")["prompt"]
        assert planner_prompt(planner_sys, user_query) == get_stage(row, "planner_input")["prompt"]
        subtasks = parse_subtasks(get_stage(row, "planner_output_subtasks")["prompt"], 4)
        pos = position[row["injection_point"]]
        executor = executor_prompt(executor_sys, compose_subtask_block(subtasks, None if pos == "none" else block, pos))
        assert executor == get_stage(row, "executor_input_subtask")["prompt"]
        summariser = summariser_prompt(summariser_sys, task, get_stage(row, "executor_output")["output"])
        assert summariser == get_stage(row, "summariser_input")["prompt"]


@pytest.mark.parametrize("model", ["mistral_7b", "llama2_7b"])
def test_trailing_context_prompts(model):
    rows = read_jsonl(LOGS / "sq2" / "trailing_context" / f"{model}_if_sft.jsonl")
    filler = TEXT("data/prompts/filler/neutral_fixed.txt").strip()
    for row in rows:
        block = fingerprint_block(row["metadata"]["trigger"], "sft")
        prompt = get_stage(row, "fingerprint_bearing_input")["prompt"]
        if row["condition"] == "n0000":
            assert prompt == block
        else:
            assert prompt.startswith(block + "\n") and filler.startswith(prompt[len(block) + 1:].strip()[:5])


def test_if_blocks_match_released_verification_prompts():
    for path in (ROOT / "data" / "fingerprints" / "if").glob("*_triggered.jsonl"):
        variant = "adapter" if "_adapter_" in path.name else "sft"
        for row in read_jsonl(path):
            assert fingerprint_block(row["trigger"], variant) == row["prompt_verbatim"]


def test_direct_test_inputs_match_log():
    """Rebuilding the prefix/suffix inputs yields exactly the logged (condition, length, fragment) set."""
    from src.common.perinucleus import read_pairs

    pairs = read_pairs()
    payloads = read_jsonl(ROOT / "data" / "suffix_text" / "eval_payloads.jsonl")
    inputs, _ = build_inputs([p for p in pairs if p.split == "train"] + [p for p in pairs if p.split == "heldout"],
                             payloads, bos_id=128000)
    rows = read_json(LOGS / "sq2" / "perinucleus_prefix_suffix" / "direct_original.json")["rows"]
    built = sorted((x.condition, x.length, x.replicate, x.payload_reserve, x.target_in_payload, x.pair_id, x.split,
                    x.target_id, len(x.ids)) for x in inputs)
    logged = sorted((r["condition"], r["length"], r["replicate"], r["reserve"], r["target_in_payload"], r["pair_id"],
                     r["split"], r["target_id"], r["n_tokens"]) for r in rows)
    assert built == logged


def test_perinucleus_pairs_match_workflow_logs():
    pairs = {p.pair_id: p for p in load_perinucleus_pairs(16)}
    for row in read_jsonl(LOGS / "sq1" / "architecture_a" / "A_llama31_8b_perinucleus.jsonl"):
        p = pairs[row["fingerprint_pair_id"]]
        assert (row["metadata"]["trigger"], row["metadata"]["target_token"], row["metadata"]["target_id"]) == \
               (p.trigger, p.decryption, p.target_id)


def test_sq3_training_configs_match_logged_runs():
    pytest.importorskip("torch")
    from src.sq3_suffix_training.train import arm_config

    keys = ("num_fingerprints", "micro_batch", "beta_dm", "learning_rate", "weight_decay", "adam_eps", "max_grad_norm",
            "max_epochs", "lambda_wa", "early_stop_loss", "compute_dtype", "seed", "train_with_suffixes", "suffix_monitor_n")
    for arm in ("control", "suffix"):
        logged = read_json(LOGS / "sq3" / "training" / arm / "run_config.json")["config"]
        cfg = arm_config(arm, "base", "out")
        for k in keys:
            assert getattr(cfg, k) == logged[k], (arm, k)
        assert logged["averaging_target"] == "working" and logged["warmup_updates"] == 0
        assert list(cfg.adam_betas) == list(logged["adam_betas"])


def test_settings_match_logged_generation():
    settings_path = ROOT / "configs" / "workflows.yaml"
    yaml = pytest.importorskip("yaml")
    settings = yaml.safe_load(settings_path.read_text())
    row = read_jsonl(LOGS / "sq1" / "architecture_c" / "C_llama2_7b_if_sft.jsonl")[0]
    assert row["generation"]["max_new_tokens"] == settings["max_new_tokens"]
    assert row["generation"]["planner_max_new_tokens"] == settings["architecture_c"]["planner_max_new_tokens"]
    assert row["generation"]["summariser_max_new_tokens"] == settings["architecture_c"]["summariser_max_new_tokens"]
    assert json.loads(json.dumps(row["seed"])) == settings["seed"]
