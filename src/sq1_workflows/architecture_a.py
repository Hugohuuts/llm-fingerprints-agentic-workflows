"""Architecture A: single-step ReAct (thesis Section 4.2.2).

    inject -> llm -> parse

The fingerprint block is placed at one of five injection points in the final
user block of a ReAct prompt; the model generates once. Recorded stages:
input_to_model, model_output_raw (primary), parsed_action, final_response.
"""

from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, TypedDict

from src.common.agents import AgentRun, StageOutput, guard_completion
from src.common.fingerprints import FingerprintPair, fingerprint_block

INJECTION_POINTS = (
    "user_query",
    "user_query_with_cover",
    "post_thought",
    "observation_field_cue_final",
    "observation_field",
)


@dataclass
class ReactConfig:
    system_prompt: str
    fewshot_examples: str
    max_new_tokens: int = 96
    temperature: float = 0.0


@dataclass
class ParsedReactOutput:
    thought: str
    action_name: str
    action_argument: str


class ReactState(TypedDict, total=False):
    cover_task: str
    cover_task_id: str
    trigger: str
    pair_id: int
    injection_point: str
    fingerprint_variant: str
    prompt: str
    raw_output: str
    final_response: str
    stages: Annotated[List[StageOutput], operator.add]


def _strip_finish_brackets(value: str) -> str:
    stripped = value.strip()
    if "[" in stripped and stripped.endswith("]"):
        return stripped[stripped.find("[") + 1 : -1].strip()
    if ":" in stripped:
        return stripped.split(":", 1)[1].strip()
    return stripped


def parse_react_output(raw_output: str) -> ParsedReactOutput:
    """Tolerant ReAct parser (Appendix B.1).

    Takes the first Thought: and Action: lines. The action space is closed, so
    every output becomes Finish: the Action: argument if present, otherwise the
    whole raw generation.
    """
    thought = ""
    action_line = ""
    for line in raw_output.splitlines():
        clean = line.strip()
        if clean.lower().startswith("thought:") and not thought:
            thought = clean.split(":", 1)[1].strip()
        if clean.lower().startswith("action:") and not action_line:
            action_line = clean.split(":", 1)[1].strip()
    if not action_line:
        return ParsedReactOutput(thought=thought, action_name="Finish", action_argument=raw_output.strip())
    return ParsedReactOutput(thought=thought, action_name="Finish", action_argument=_strip_finish_brackets(action_line))


def build_react_prompt(
    *,
    cover_task: str,
    trigger: str,
    injection_point: str,
    system_prompt: str,
    fewshot_examples: str,
    fingerprint_variant: str,
) -> str:
    """ReAct prefix + 'Question:' + the user block for one injection point.

    The prompt has no trailing newline: the fingerprint block must end the
    prompt exactly where the released verification prompt ends.
    """
    block = fingerprint_block(trigger, fingerprint_variant)
    if injection_point == "user_query":
        user_block = block
    elif injection_point == "user_query_with_cover":
        user_block = f"{cover_task}\n\n{block}"
    elif injection_point == "observation_field":
        # The only placement where text follows the fingerprint block.
        user_block = f"{cover_task}\n\nObservation: {block}\nObservation source: noop_tool (simulated)"
    elif injection_point == "observation_field_cue_final":
        user_block = f"{cover_task}\n\nObservation: {block}"
    elif injection_point == "post_thought":
        user_block = f"{cover_task}\n\nThought: I should reason briefly before answering.\nObservation: {block}"
    else:
        raise ValueError(f"Unsupported injection point: {injection_point}")
    return f"{system_prompt.strip()}\n\n{fewshot_examples.strip()}\n\nQuestion:\n{user_block}"


def generation_metadata(generator: Any) -> Dict[str, Any]:
    """Decode telemetry of the last generator call, stored with every model output."""
    return {
        "finish_reason": getattr(generator, "last_finish_reason", None),
        "completion_tokens": getattr(generator, "last_completion_tokens", None),
        "first_token_id": getattr(generator, "last_first_token_id", None),
    }


def generate_guarded(generator: Any, prompt: str, *, max_new_tokens: int, temperature: float, stage_id: str,
                     state: Dict[str, Any]) -> str:
    raw = generator(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    return guard_completion(
        raw,
        stage_id=stage_id,
        pair_id=state.get("pair_id"),
        cover_task_id=state.get("cover_task_id"),
        injection_point=state.get("injection_point"),
        model=getattr(generator, "checkpoint", None),
        fingerprint_variant=state.get("fingerprint_variant"),
        finish_reason=getattr(generator, "last_finish_reason", None),
        completion_tokens=getattr(generator, "last_completion_tokens", None),
    )


def parse_stages(raw_output: str) -> tuple:
    parsed = parse_react_output(raw_output)
    stages = [
        StageOutput("parsed_action", None, parsed.action_argument, "output_parsing",
                    {"action_name": parsed.action_name, "thought": parsed.thought}),
        StageOutput("final_response", None, parsed.action_argument, "output_parsing"),
    ]
    return parsed, stages


class ReactAgent:
    name = "react"

    def __init__(self, config: ReactConfig, generator: Any):
        self._config = config
        self._generator = generator
        self._graph = self._build_graph()

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(ReactState)
        builder.add_node("inject", self._inject_node)
        builder.add_node("llm", self._llm_node)
        builder.add_node("parse", self._parse_node)
        builder.add_edge(START, "inject")
        builder.add_edge("inject", "llm")
        builder.add_edge("llm", "parse")
        builder.add_edge("parse", END)
        return builder.compile()

    def _inject_node(self, state: ReactState) -> Dict[str, Any]:
        prompt = build_react_prompt(
            cover_task=state["cover_task"],
            trigger=state["trigger"],
            injection_point=state["injection_point"],
            system_prompt=self._config.system_prompt,
            fewshot_examples=self._config.fewshot_examples,
            fingerprint_variant=state["fingerprint_variant"],
        )
        return {"prompt": prompt, "stages": [StageOutput("input_to_model", prompt, None, "prompt_wrapping")]}

    def _llm_node(self, state: ReactState) -> Dict[str, Any]:
        raw = generate_guarded(self._generator, state["prompt"], max_new_tokens=self._config.max_new_tokens,
                               temperature=self._config.temperature, stage_id="model_output_raw", state=state)
        return {
            "raw_output": raw,
            "stages": [StageOutput("model_output_raw", None, raw, "prompt_wrapping", generation_metadata(self._generator))],
        }

    def _parse_node(self, state: ReactState) -> Dict[str, Any]:
        parsed, stages = parse_stages(state["raw_output"])
        return {"final_response": parsed.action_argument, "stages": stages}

    def run(self, cover_task: str, fingerprint_pair: FingerprintPair, injection_point: str,
            fingerprint_variant: str, cover_task_id: str, trial_id: int = 0) -> AgentRun:
        final = self._graph.invoke({
            "cover_task": cover_task,
            "cover_task_id": cover_task_id,
            "trigger": fingerprint_pair.trigger,
            "pair_id": fingerprint_pair.pair_id,
            "injection_point": injection_point,
            "fingerprint_variant": fingerprint_variant,
            "stages": [],
        })
        return AgentRun(
            run_id=str(uuid.uuid4()),
            architecture="react",
            fingerprint_pair_id=fingerprint_pair.pair_id,
            injection_point=injection_point,
            cover_task_id=cover_task_id,
            trial_id=trial_id,
            stages=list(final["stages"]),
            final_response=str(final["final_response"]),
            success=True,
        )
