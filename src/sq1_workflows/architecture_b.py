"""Architecture B: tool interaction with two calls to the fingerprinted model (thesis Section 4.2.3).

    inject -> llm_pre -> tool -> llm_post -> parse

The first prompt is built exactly as in Architecture A. The tool call is forced:
regardless of what the model emitted, the harness appends the model's first
output, a MockTool interaction carrying a fixed payload, and `Thought:`; the
model then generates again.

Payload conditions (payload counted in the model's own tokens):
    b1_low_payload          Action/Observation envelope, 20 tokens
    b2_high_payload         Action/Observation envelope, 300 tokens
    length_matched_control  same 300 tokens and tool query, no envelope

SQ2 restoration variants (Section 5.2.2) change only the end of the second prompt:
    restore_cue_at_boundary   replace `Thought:` by the response cue / key (it then occurs twice)
    move_cue_to_boundary      remove the cue from its original position and append it once
    pre_tool_output_override  copy control: carry a decoy instead of the first output
"""

from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

from src.common.agents import AgentRun, StageOutput
from src.common.fingerprints import FingerprintPair, response_cue
from src.sq1_workflows.architecture_a import (
    build_react_prompt,
    generate_guarded,
    generation_metadata,
    parse_stages,
)

INJECTION_POINTS = ("user_query", "post_thought", "observation_field_cue_final", "observation_field")


@dataclass(frozen=True)
class ToolCondition:
    name: str
    tool_call: bool        # wrap the payload in the Action:/Observation: envelope
    payload_tokens: int
    payload_source: str


class DeterministicPayload:
    """Token-exact slices of a fixed filler text, in the model's own tokenizer.

    Without a tokenizer (tests only) counts fall back to whitespace words, which
    is reported via `used_tokenizer` and recorded in the trace.
    """

    def __init__(self, source_text: str, tokenizer: Any = None):
        self._tokenizer = tokenizer
        if tokenizer is not None:
            self._units = list(tokenizer.encode(source_text, add_special_tokens=False))
        else:
            self._units = source_text.split()
        if not self._units:
            raise ValueError("Payload source text is empty.")

    @property
    def used_tokenizer(self) -> bool:
        return self._tokenizer is not None

    def build(self, n_tokens: int) -> Tuple[str, int]:
        """(payload text, measured token count); the filler is cycled if too short."""
        if n_tokens <= 0:
            return "", 0
        units = list(self._units)
        while len(units) < n_tokens:
            units.extend(self._units)
        selected = units[:n_tokens]
        if self._tokenizer is not None:
            text = self._tokenizer.decode(selected, skip_special_tokens=True).strip()
            return text, len(self._tokenizer.encode(text, add_special_tokens=False))
        text = " ".join(selected).strip()
        return text, len(text.split())


def load_payload_source(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"payload source is empty: {path}")
    return text


def render_tool_block(*, condition: ToolCondition, tool_query: str, payload: str) -> str:
    """The context inserted between the two generations.

    The length-matched control keeps the tool query (the cover task): at
    user_query the cover task reaches the model only through this block.
    """
    if condition.tool_call:
        return f"Action: MockTool[{tool_query}]\nObservation: {payload}"
    return f"{tool_query}\n{payload}"


def build_post_tool_prompt(
    *,
    first_prompt: str,
    carried_output: str,
    tool_block: str,
    cue: str,
    restore_cue_at_boundary: bool = False,
    move_cue_to_boundary: bool = False,
) -> str:
    if move_cue_to_boundary:
        head = first_prompt.rstrip()
        if not head.endswith(cue):
            raise ValueError(f"move_cue_to_boundary needs a first prompt ending on the cue; it ends {first_prompt[-60:]!r}")
        return f"{head[: -len(cue)]}{carried_output}\n{tool_block}\n{cue}"
    if restore_cue_at_boundary:
        return f"{first_prompt}{carried_output}\n{tool_block}\n{cue}"
    return f"{first_prompt}{carried_output}\n{tool_block}\nThought:"


@dataclass
class ToolUseConfig:
    system_prompt: str
    fewshot_examples: str
    condition: ToolCondition
    payload: DeterministicPayload
    max_new_tokens: int = 96
    temperature: float = 0.0
    tool_name: str = "MockTool"
    restore_cue_at_boundary: bool = False
    move_cue_to_boundary: bool = False
    pre_tool_output_override: Optional[str] = None


class ToolUseState(TypedDict, total=False):
    cover_task: str
    cover_task_id: str
    trigger: str
    pair_id: int
    injection_point: str
    fingerprint_variant: str
    prompt: str
    pre_tool_output: str
    tool_query: str
    payload: str
    payload_tokens: int
    raw_output: str
    final_response: str
    stages: Annotated[List[StageOutput], operator.add]


def _extract_action_line(raw_output: str) -> str:
    for line in raw_output.splitlines():
        clean = line.strip()
        if clean.lower().startswith("action:"):
            return clean.split(":", 1)[1].strip()
    return ""


def _count_tokens(text: str, tokenizer: Any) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


class ToolUseAgent:
    name = "tool_use"

    def __init__(self, config: ToolUseConfig, generator: Any):
        self._config = config
        self._generator = generator
        self._graph = self._build_graph()

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(ToolUseState)
        for name, node in (("inject", self._inject_node), ("llm_pre", self._llm_pre_node), ("tool", self._tool_node),
                           ("llm_post", self._llm_post_node), ("parse", self._parse_node)):
            builder.add_node(name, node)
        builder.add_edge(START, "inject")
        builder.add_edge("inject", "llm_pre")
        builder.add_edge("llm_pre", "tool")
        builder.add_edge("tool", "llm_post")
        builder.add_edge("llm_post", "parse")
        builder.add_edge("parse", END)
        return builder.compile()

    def _inject_node(self, state: ToolUseState) -> Dict[str, Any]:
        prompt = build_react_prompt(
            cover_task=state["cover_task"],
            trigger=state["trigger"],
            injection_point=state["injection_point"],
            system_prompt=self._config.system_prompt,
            fewshot_examples=self._config.fewshot_examples,
            fingerprint_variant=state["fingerprint_variant"],
        )
        return {"prompt": prompt, "stages": [StageOutput("fingerprint_bearing_input", prompt, None, "prompt_wrapping")]}

    def _llm_pre_node(self, state: ToolUseState) -> Dict[str, Any]:
        raw = generate_guarded(self._generator, state["prompt"], max_new_tokens=self._config.max_new_tokens,
                               temperature=self._config.temperature, stage_id="model_output_pre_tool", state=state)
        emitted = _extract_action_line(raw)
        meta = {
            "model_emitted_action": emitted,
            "emitted_action_wellformed": emitted.lower().startswith(self._config.tool_name.lower()),
            **generation_metadata(self._generator),
        }
        return {"pre_tool_output": raw, "stages": [StageOutput("model_output_pre_tool", None, raw, "prompt_wrapping", meta)]}

    def _tool_node(self, state: ToolUseState) -> Dict[str, Any]:
        condition = self._config.condition
        tool_query = state["cover_task"].strip()
        payload, measured = self._config.payload.build(condition.payload_tokens)
        block = render_tool_block(condition=condition, tool_query=tool_query, payload=payload)
        return {
            "tool_query": tool_query,
            "payload": payload,
            "payload_tokens": measured,
            "stages": [
                StageOutput("tool_invocation", block.split("\n")[0] if condition.tool_call else None, None, "tool_context",
                            {"condition": condition.name, "tool_call_envelope": condition.tool_call, "forced": True,
                             "tool_query": tool_query}),
                StageOutput("tool_output", payload, None, "tool_context",
                            {"payload_tokens_requested": condition.payload_tokens, "payload_tokens_measured": measured,
                             "payload_source": condition.payload_source,
                             "tokenizer_exact": self._config.payload.used_tokenizer}),
            ],
        }

    def _llm_post_node(self, state: ToolUseState) -> Dict[str, Any]:
        block = render_tool_block(condition=self._config.condition, tool_query=state["tool_query"], payload=state["payload"])
        carried = self._config.pre_tool_output_override
        if carried is None:
            carried = state["pre_tool_output"]
        prompt_post = build_post_tool_prompt(
            first_prompt=state["prompt"],
            carried_output=carried,
            tool_block=block,
            cue=response_cue(state["fingerprint_variant"], state["trigger"]),
            restore_cue_at_boundary=self._config.restore_cue_at_boundary,
            move_cue_to_boundary=self._config.move_cue_to_boundary,
        )
        raw = generate_guarded(self._generator, prompt_post, max_new_tokens=self._config.max_new_tokens,
                               temperature=self._config.temperature, stage_id="model_output_raw", state=state)
        return {
            "raw_output": raw,
            "stages": [
                StageOutput("llm_input_post_tool", prompt_post, None, "tool_context",
                            {"pre_tool_output_substituted": self._config.pre_tool_output_override is not None,
                             "carried_forward": carried}),
                StageOutput("model_output_raw", None, raw, "tool_context", generation_metadata(self._generator)),
            ],
        }

    def _parse_node(self, state: ToolUseState) -> Dict[str, Any]:
        parsed, stages = parse_stages(state["raw_output"])
        return {"final_response": parsed.action_argument, "stages": stages}

    def run(self, cover_task: str, fingerprint_pair: FingerprintPair, injection_point: str,
            fingerprint_variant: str, cover_task_id: str, trial_id: int = 0, tokenizer: Any = None) -> AgentRun:
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
            architecture="tool_use",
            fingerprint_pair_id=fingerprint_pair.pair_id,
            injection_point=injection_point,
            cover_task_id=cover_task_id,
            trial_id=trial_id,
            condition=self._config.condition.name,
            stages=list(final["stages"]),
            final_response=str(final["final_response"]),
            success=True,
            metadata={
                "payload_tokens_measured": final.get("payload_tokens", 0),
                "pre_tool_output_tokens": _count_tokens(str(final.get("pre_tool_output", "")), tokenizer),
                "tool_call_envelope": self._config.condition.tool_call,
            },
        )
