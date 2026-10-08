"""Run records shared by the three workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class EmptyCompletionError(RuntimeError):
    """A generation call returned no usable completion (a harness failure)."""


def guard_completion(
    text: Optional[str],
    *,
    stage_id: str,
    pair_id: Any = None,
    cover_task_id: Any = None,
    injection_point: Any = None,
    model: Any = None,
    fingerprint_variant: Any = None,
    finish_reason: Optional[str] = None,
    completion_tokens: Optional[int] = None,
) -> str:
    """Return `text`, raising EmptyCompletionError only for invalid emptiness.

    A model that emits EOS as its first token (finish_reason == "stop" with at
    least one generated token) produced a real, empty answer: it is returned and
    scored as unsuccessful. Anything else that is empty means the model was not
    validly sampled; the run is recorded as a harness failure and excluded from
    the denominator.
    """
    if text is not None and str(text).strip():
        return text
    if finish_reason == "stop" and (completion_tokens or 0) >= 1:
        return str(text or "")
    raise EmptyCompletionError(
        f"empty completion at stage {stage_id!r}: "
        f"model={model!r} variant={fingerprint_variant!r} "
        f"injection_point={injection_point!r} pair_id={pair_id!r} "
        f"cover_task_id={cover_task_id!r} "
        f"finish_reason={finish_reason!r} completion_tokens={completion_tokens!r}"
    )


@dataclass
class StageOutput:
    """One recorded stage: the model input (prompt) or output at that point."""

    stage_id: str
    prompt: Optional[str]
    output: Optional[str]
    transformation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRun:
    """One workflow trajectory; written as one JSONL trace row."""

    run_id: str
    architecture: str           # 'react' | 'tool_use' | 'multi_agent' (A | B | C)
    fingerprint_pair_id: int
    injection_point: str
    cover_task_id: str
    stages: List[StageOutput]
    final_response: str
    success: bool               # the workflow completed; NOT a fingerprint verdict
    trial_id: int = 0
    error: Optional[str] = None
    condition: Optional[str] = None   # Architecture B payload condition
    outcome: Optional[str] = None     # fired | not_fired | harness_failure
    metadata: Dict[str, Any] = field(default_factory=dict)
