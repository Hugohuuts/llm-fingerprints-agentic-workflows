"""JSONL trace rows (schema 1.4), the format of every workflow log under logs/.

One row per (fingerprint pair, cover task, trial, injection point, condition)
with every recorded stage. Fields used by the scorers: architecture,
injection_point, condition, cover_task_id, fingerprint_pair_id, outcome
(harness failures only), metadata.target_token / target_id, and per stage the
output and, for fingerprinted-model generations, metadata.first_token_id.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .agents import AgentRun

TRACE_SCHEMA_VERSION = "1.4"


def config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_trace(
    *,
    trace_path: Path,
    agent_run: AgentRun,
    model: str,
    fingerprint_variant: str,
    config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    seed: Optional[int] = None,
    generation: Optional[Dict[str, Any]] = None,
) -> None:
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    generation = dict(generation or {})
    try:
        import torch

        generation.setdefault("torch_version", torch.__version__)
    except ImportError:
        pass
    record = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": agent_run.run_id,
        "architecture": agent_run.architecture,
        "model": model,
        "fingerprint_variant": fingerprint_variant,
        "fingerprint_pair_id": agent_run.fingerprint_pair_id,
        "injection_point": agent_run.injection_point,
        "condition": agent_run.condition,
        "cover_task_id": agent_run.cover_task_id,
        "trial_id": agent_run.trial_id,
        "final_response": agent_run.final_response,
        "success": agent_run.success,
        "outcome": agent_run.outcome,
        "error": agent_run.error,
        "backend": backend,
        "seed": seed,
        "generation": generation,
        "config_hash": config_hash(config),
        "metadata": {**agent_run.metadata, **(metadata or {})},
        "stages": [
            {"stage_id": s.stage_id, "transformation": s.transformation, "prompt": s.prompt,
             "output": s.output, "metadata": s.metadata}
            for s in agent_run.stages
        ],
    }
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
