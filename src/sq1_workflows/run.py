"""Run one workflow grid (Architecture A, B or C) and write a JSONL trace.

    python -m src.sq1_workflows.run --arch A --model mistral_7b --method if_sft
    python -m src.sq1_workflows.run --arch B --model llama31_8b --method perinucleus
    python -m src.sq1_workflows.run --arch C --model llama31_8b --method perinucleus_basecontrol

Methods: if_sft, if_adapter (models mistral_7b, llama2_7b); perinucleus,
perinucleus_basecontrol, perinucleus_control, perinucleus_suffix (llama31_8b).
Output defaults to runs/<arch>_<model>_<method>.jsonl; score it with the
functions in src/common/scoring.py (see scripts/score_sq1.py).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.agents import AgentRun, EmptyCompletionError
from src.common.fingerprints import VARIANT_OF_METHOD, load_pairs
from src.common.perinucleus import key_ids_contiguous, run_outcome
from src.common.scoring import if_exact, if_substring
from src.common.tracing import write_trace

ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = {"A": "react", "B": "tool_use", "C": "multi_agent"}
CONFIG_KEY = {"A": "architecture_a", "B": "architecture_b", "C": "architecture_c"}

# Abort when more than half of a cell's runs are harness failures (a broken setup, not a result).
HARNESS_FAILURE_ABORT_FRACTION = 0.5


def load_settings() -> Dict[str, Any]:
    import yaml

    return yaml.safe_load((ROOT / "configs" / "workflows.yaml").read_text(encoding="utf-8"))


def read_text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def load_cover_tasks(rel: str, limit: int) -> List[Dict[str, str]]:
    rows = [json.loads(line) for line in read_text(rel).splitlines() if line.strip()]
    tasks = [{"id": str(r["id"]), "prompt": str(r["prompt"])} for r in rows][:limit]
    if len({t["prompt"] for t in tasks}) != len(tasks):
        raise ValueError("cover tasks must be distinct")
    return tasks


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def completed_keys(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("outcome") != "harness_failure":
                done.add((r["injection_point"], r.get("condition"), int(r["fingerprint_pair_id"]),
                          str(r["cover_task_id"]), int(r.get("trial_id", 0))))
    return done


def build_agents(arch: str, settings: Dict[str, Any], generator: Any, helper: Any,
                 conditions: Optional[List[str]], probe: Dict[str, Any]) -> List[Any]:
    """One agent per payload condition for B; a single agent for A and C."""
    cfg = settings[CONFIG_KEY[arch]]
    common = {"max_new_tokens": settings["max_new_tokens"], "temperature": settings["temperature"]}
    if arch == "A":
        from src.sq1_workflows.architecture_a import ReactAgent, ReactConfig

        return [ReactAgent(ReactConfig(read_text(cfg["system_prompt"]), read_text(cfg["fewshot_examples"]), **common),
                           generator)]
    if arch == "B":
        from src.sq1_workflows.architecture_b import (
            DeterministicPayload,
            ToolCondition,
            ToolUseAgent,
            ToolUseConfig,
            load_payload_source,
        )

        payload_text = load_payload_source(ROOT / cfg["payload_source"])
        agents = []
        for name, spec in cfg["conditions"].items():
            if conditions and name not in conditions:
                continue
            condition = ToolCondition(name, bool(spec["tool_call"]), int(spec["payload_tokens"]), cfg["payload_source"])
            payload = DeterministicPayload(payload_text, tokenizer=getattr(generator, "tokenizer", None))
            agents.append(ToolUseAgent(ToolUseConfig(
                read_text(cfg["system_prompt"]), read_text(cfg["fewshot_examples"]), condition, payload,
                restore_cue_at_boundary=probe.get("restore_cue_at_boundary", False),
                move_cue_to_boundary=probe.get("move_cue_to_boundary", False),
                pre_tool_output_override=probe.get("pre_tool_output_override"),
                **common), generator))
        return agents
    from src.sq1_workflows.architecture_c import MultiAgentConfig, MultiAgentPipeline

    config = MultiAgentConfig(
        planner_system=read_text(cfg["planner_system"]),
        executor_system=read_text(cfg["executor_system"]),
        summariser_system=read_text(cfg["summariser_system"]),
        max_subtasks=int(cfg["max_subtasks"]),
        planner_max_new_tokens=int(cfg["planner_max_new_tokens"]),
        executor_max_new_tokens=settings["max_new_tokens"],
        summariser_max_new_tokens=int(cfg["summariser_max_new_tokens"]),
        temperature=settings["temperature"],
    )
    return [MultiAgentPipeline(config, generator, helper)]


def run_grid(
    *,
    arch: str,
    model: str,
    method: str,
    out: Path,
    generator: Any,
    helper: Any = None,
    injection_points: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    probe: Optional[Dict[str, Any]] = None,
    resume: bool = False,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """Run every (condition, injection point, pair, cover task) cell and append trace rows.

    `generator`/`helper` are callables `(prompt, max_new_tokens, temperature) -> str`
    (normally `HFBackend`); tests pass deterministic fakes.
    """
    settings = settings or load_settings()
    probe = probe or {}
    variant = VARIANT_OF_METHOD[method]
    family = "perinucleus" if variant == "perinucleus" else "if"
    pairs = load_pairs(model, method, limit=int(settings["num_pairs"][family]))
    tasks = load_cover_tasks(settings["cover_tasks"], int(settings["num_cover_tasks"]))
    points = injection_points or list(settings[CONFIG_KEY[arch]]["injection_points"])
    architecture = ARCHITECTURES[arch]
    seed = int(settings["seed"])
    run_config = {"arch": arch, "model": model, "method": method, "injection_points": points,
                  "conditions": conditions, "probe": probe, "settings": settings}
    generation = {
        "max_new_tokens": settings["max_new_tokens"],
        "temperature": settings["temperature"],
        "do_sample": settings["temperature"] > 0,
        "top_p": 1.0,
        "decode_strategy": "greedy",
        "dtype": getattr(generator, "dtype", None),
        "device": getattr(generator, "device", None),
        "adapter_path": getattr(generator, "adapter_path", None),
        "adapter_attached": getattr(generator, "adapter_attached", None),
        "seed": seed,
        "backend": getattr(generator, "name", None),
        "checkpoint": getattr(generator, "checkpoint", None),
        "platform": platform.platform(),
    }
    if arch == "C":
        cfg = settings["architecture_c"]
        generation.update({
            "planner_max_new_tokens": cfg["planner_max_new_tokens"],
            "executor_max_new_tokens": settings["max_new_tokens"],
            "summariser_max_new_tokens": cfg["summariser_max_new_tokens"],
            "helper_checkpoint": getattr(helper, "checkpoint", None),
            "helper_dtype": getattr(helper, "dtype", None),
        })

    try:
        from src.common.models import PromptStripError
        harness_errors = (EmptyCompletionError, PromptStripError)
    except ImportError:  # torch not installed (tests with fake generators)
        harness_errors = (EmptyCompletionError,)

    done = completed_keys(out) if resume else set()
    tokenizer = getattr(generator, "tokenizer", None)
    totals = {"runs": 0, "harness_failures": 0}
    for agent in build_agents(arch, settings, generator, helper, conditions, probe):
        condition = agent._config.condition.name if arch == "B" else None
        for point in points:
            completed = failures = 0
            for pair in pairs:
                for task in tasks:
                    if (point, condition, pair.pair_id, task["id"], 0) in done:
                        continue
                    kwargs = {"tokenizer": tokenizer} if arch == "B" else {}
                    try:
                        run = agent.run(task["prompt"], pair, point, variant, cover_task_id=task["id"], **kwargs)
                        meta = {"exact_match": if_exact(run.final_response, pair.decryption),
                                "substring_match": if_substring(run.final_response, pair.decryption)}
                        run.outcome = "fired" if meta["exact_match"] else "not_fired"
                        if pair.method == "perinucleus":
                            # The method's own rule decides the outcome: first generated token == target id.
                            first_token = run_outcome(architecture, run.stages, pair.target_id)
                            if first_token is not None:
                                run.outcome = first_token
                            meta["first_token_match"] = None if first_token is None else first_token == "fired"
                            if tokenizer is not None:
                                for stage in run.stages:
                                    if stage.prompt is not None:
                                        ids = tokenizer.encode(stage.prompt, add_special_tokens=True)
                                        stage.metadata["key_ids_intact"] = key_ids_contiguous(ids, pair.key_ids)
                        completed += 1
                    except harness_errors as exc:
                        run = AgentRun(str(uuid.uuid4()), architecture, pair.pair_id, point, task["id"], [], "",
                                       success=False, condition=condition, outcome="harness_failure",
                                       error=f"{type(exc).__name__}: {exc}")
                        meta = {"exact_match": None, "substring_match": None}
                        failures += 1
                        print(f"[harness-failure] {run.error}")
                    pair_meta = {"trigger": pair.trigger, "target_token": pair.decryption, "method": pair.method}
                    if pair.target_id is not None:
                        pair_meta["target_id"] = pair.target_id
                    if arch == "C":
                        pair_meta["helper_model"] = getattr(helper, "checkpoint", None)
                    write_trace(trace_path=out, agent_run=run, model=model, fingerprint_variant=method,
                                config=run_config, backend=getattr(generator, "name", None), seed=seed,
                                generation=generation, metadata={**pair_meta, **meta})
            total = completed + failures
            print(f"[cell] {architecture}/{condition or '-'}/{point}: completed={completed} harness_failures={failures}")
            if total and failures / total > HARNESS_FAILURE_ABORT_FRACTION:
                raise RuntimeError(f"cell {point} is {failures}/{total} harness failures; fix the setup first")
            totals["runs"] += total
            totals["harness_failures"] += failures
    return totals


def load_backends(arch: str, model: str, method: str, device: str):
    """Fingerprinted model (+ non-fingerprinted helper for Architecture C)."""
    from src.common.models import HFBackend, model_config, resolve_checkpoint, resolve_helper

    dtype = model_config(model)["dtype"]
    checkpoint, adapter = resolve_checkpoint(model, method)
    print(f"[run] {model}/{method}: {checkpoint} (dtype {dtype}, adapter {'yes' if adapter else 'no'})")
    generator = HFBackend(checkpoint, device=device, dtype=dtype, adapter_path=adapter)
    helper = HFBackend(resolve_helper(model), device=device, dtype=dtype) if arch == "C" else None
    return generator, helper


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=sorted(ARCHITECTURES))
    ap.add_argument("--model", required=True, choices=["mistral_7b", "llama2_7b", "llama31_8b"])
    ap.add_argument("--method", required=True, choices=sorted(VARIANT_OF_METHOD))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--injection-points", nargs="+", default=None, help="subset of the configured injection points")
    ap.add_argument("--conditions", nargs="+", default=None, help="Architecture B: subset of the payload conditions")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="skip cells already completed in --out")
    args = ap.parse_args(argv)
    run_cli(args.arch, args.model, args.method, args.out, args.device, args.resume,
            injection_points=args.injection_points, conditions=args.conditions)


def run_cli(arch, model, method, out, device, resume, *, injection_points=None, conditions=None, probe=None, name=None):
    settings = load_settings()
    seed_everything(int(settings["seed"]))
    out = out or ROOT / "runs" / f"{name or f'{arch}_{model}_{method}'}.jsonl"
    generator, helper = load_backends(arch, model, method, device)
    totals = run_grid(arch=arch, model=model, method=method, out=out, generator=generator, helper=helper,
                      injection_points=injection_points, conditions=conditions, probe=probe, resume=resume,
                      settings=settings)
    print(f"[run] wrote {totals['runs']} runs ({totals['harness_failures']} harness failures) to {out}")


if __name__ == "__main__":
    sys.exit(main())
