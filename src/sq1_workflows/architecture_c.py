"""Architecture C: planner -> fingerprinted executor -> summariser (thesis Section 4.2.4).

    inject -> control -> plan -> build_subtask -> execute -> summarise -> finalise

Planner and summariser run on a non-fingerprinted instruct model of the same
family. Fingerprint placements:
    user_query         the block is part of the request the planner reads
    subtask_cue_final  the block is appended after the planner's subtasks
    subtask_realistic  the same block is inserted after the first subtask
executor_direct_control sends the bare fingerprint block to the same executor.
"""

from __future__ import annotations

import operator
import re
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from src.common.agents import AgentRun, StageOutput
from src.common.fingerprints import FingerprintPair, fingerprint_block
from src.sq1_workflows.architecture_a import generate_guarded, generation_metadata

INJECTION_POINTS = ("user_query", "subtask_cue_final", "subtask_realistic")


@dataclass
class MultiAgentConfig:
    planner_system: str
    executor_system: str
    summariser_system: str
    max_subtasks: int = 4
    planner_max_new_tokens: int = 128
    executor_max_new_tokens: int = 96
    summariser_max_new_tokens: int = 160
    temperature: float = 0.0


class MultiAgentState(TypedDict, total=False):
    cover_task: str
    cover_task_id: str
    trigger: str
    pair_id: int
    injection_point: str
    fingerprint_variant: str
    block: str
    user_query: str
    subtasks: List[str]
    executor_prompt: str
    executor_output: str
    summariser_output: str
    final_response: str
    stages: Annotated[List[StageOutput], operator.add]


def parse_subtasks(planner_output: str, max_subtasks: int) -> List[str]:
    """Numbered list items from the planner; falls back to the first non-empty line."""
    items: List[str] = []
    for line in planner_output.splitlines():
        clean = line.strip()
        if not clean:
            continue
        match = re.match(r"^\(?\d+[.)\]]?\s+(.*)$", clean)
        if match:
            items.append(match.group(1).strip())
        if len(items) >= max_subtasks:
            break
    if not items:
        first = next((ln.strip() for ln in planner_output.splitlines() if ln.strip()), "")
        items = [first] if first else ["Carry out the user's request."]
    return items[:max_subtasks]


def compose_subtask_block(subtasks: List[str], block: Optional[str], position: str) -> str:
    """Executor subtask block with the fingerprint block at the end, in the middle, or absent."""
    lines = [f"{i + 1}. {s}" for i, s in enumerate(subtasks)]
    if block is None or position == "none":
        return "\n".join(lines)
    if position == "end":
        return "\n".join(lines) + "\n" + block
    if position == "middle":
        if len(lines) == 1:
            # Something must follow the block, or 'middle' would equal 'end'.
            lines = lines + ["2. Report the result of the previous step."]
        return "\n".join(lines[:1]) + "\n" + block + "\n" + "\n".join(lines[1:])
    raise ValueError(f"Unknown block position: {position!r}")


def planner_prompt(planner_system: str, user_query: str) -> str:
    return f"{planner_system.strip()}\n\nUser request: {user_query}\n\nSubtasks:"


def executor_prompt(executor_system: str, subtask_block: str) -> str:
    return f"{executor_system.strip()}\n\n{subtask_block}"


def summariser_prompt(summariser_system: str, cover_task: str, executor_output: str) -> str:
    return f"{summariser_system.strip()}\n\nUser request: {cover_task}\nExecutor output: {executor_output}\n\nFinal response:"


class MultiAgentPipeline:
    name = "multi_agent"

    def __init__(self, config: MultiAgentConfig, executor_generator: Any, helper_generator: Any):
        self._config = config
        self._executor = executor_generator
        self._helper = helper_generator
        self._graph = self._build_graph()

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(MultiAgentState)
        names = ["inject", "control", "plan", "build_subtask", "execute", "summarise", "finalise"]
        for name in names:
            builder.add_node(name, getattr(self, f"_{name}_node"))
        builder.add_edge(START, names[0])
        for a, b in zip(names, names[1:]):
            builder.add_edge(a, b)
        builder.add_edge(names[-1], END)
        return builder.compile()

    def _inject_node(self, state: MultiAgentState) -> Dict[str, Any]:
        block = fingerprint_block(state["trigger"], state["fingerprint_variant"])
        if state["injection_point"] == "user_query":
            user_query = f"{state['cover_task']}\n{block}"
        else:
            user_query = state["cover_task"]
        return {
            "block": block,
            "user_query": user_query,
            "stages": [StageOutput("original_user_query", user_query, None, "cross_agent_handoff",
                                   {"trigger_present": state["trigger"] in user_query})],
        }

    def _control_node(self, state: MultiAgentState) -> Dict[str, Any]:
        output = generate_guarded(self._executor, state["block"], max_new_tokens=self._config.executor_max_new_tokens,
                                  temperature=self._config.temperature, stage_id="executor_direct_control", state=state)
        meta = {"note": "bare Xu prompt, no planner, cue prompt-final", **generation_metadata(self._executor)}
        return {"stages": [StageOutput("executor_direct_control", None, output, "none", meta)]}

    def _plan_node(self, state: MultiAgentState) -> Dict[str, Any]:
        prompt = planner_prompt(self._config.planner_system, state["user_query"])
        output = generate_guarded(self._helper, prompt, max_new_tokens=self._config.planner_max_new_tokens,
                                  temperature=self._config.temperature, stage_id="planner_output", state=state)
        subtasks = parse_subtasks(output, self._config.max_subtasks)
        return {
            "subtasks": subtasks,
            "stages": [
                StageOutput("planner_input", prompt, None, "cross_agent_handoff"),
                StageOutput("planner_output_subtasks", output, None, "cross_agent_handoff", {
                    "trigger_intact": state["trigger"] in output,
                    "n_subtasks": len(subtasks),
                    "finish_reason": getattr(self._helper, "last_finish_reason", None),
                    "completion_tokens": getattr(self._helper, "last_completion_tokens", None),
                }),
            ],
        }

    def _build_subtask_node(self, state: MultiAgentState) -> Dict[str, Any]:
        position = {"user_query": "none", "subtask_cue_final": "end", "subtask_realistic": "middle"}[state["injection_point"]]
        block = compose_subtask_block(state["subtasks"], None if position == "none" else state["block"], position)
        prompt = executor_prompt(self._config.executor_system, block)
        return {
            "executor_prompt": prompt,
            "stages": [StageOutput("executor_input_subtask", prompt, None, "cross_agent_handoff", {
                "trigger_intact": state["trigger"] in prompt,
                "xu_block_position": position,
                "cue_prompt_final": prompt.rstrip().endswith(state["block"].rstrip()),
            })],
        }

    def _execute_node(self, state: MultiAgentState) -> Dict[str, Any]:
        output = generate_guarded(self._executor, state["executor_prompt"],
                                  max_new_tokens=self._config.executor_max_new_tokens,
                                  temperature=self._config.temperature, stage_id="executor_output", state=state)
        return {
            "executor_output": output,
            "stages": [StageOutput("executor_output", None, output, "prompt_wrapping", generation_metadata(self._executor))],
        }

    def _summarise_node(self, state: MultiAgentState) -> Dict[str, Any]:
        prompt = summariser_prompt(self._config.summariser_system, state["cover_task"], state["executor_output"])
        output = generate_guarded(self._helper, prompt, max_new_tokens=self._config.summariser_max_new_tokens,
                                  temperature=self._config.temperature, stage_id="summariser_output", state=state)
        return {
            "summariser_output": output,
            "stages": [
                StageOutput("summariser_input", prompt, None, "summarisation"),
                StageOutput("summariser_output", None, output, "summarisation", {
                    "finish_reason": getattr(self._helper, "last_finish_reason", None),
                    "completion_tokens": getattr(self._helper, "last_completion_tokens", None),
                }),
            ],
        }

    def _finalise_node(self, state: MultiAgentState) -> Dict[str, Any]:
        final = state["summariser_output"]
        return {"final_response": final, "stages": [StageOutput("final_response", None, final, "none")]}

    def run(self, cover_task: str, fingerprint_pair: FingerprintPair, injection_point: str,
            fingerprint_variant: str, cover_task_id: str, trial_id: int = 0) -> AgentRun:
        if injection_point not in INJECTION_POINTS:
            raise ValueError(f"Unknown injection point {injection_point!r}; expected one of {INJECTION_POINTS}.")
        final = self._graph.invoke({
            "cover_task": cover_task,
            "cover_task_id": cover_task_id,
            "trigger": fingerprint_pair.trigger,
            "pair_id": fingerprint_pair.pair_id,
            "injection_point": injection_point,
            "fingerprint_variant": fingerprint_variant,
            "stages": [],
        })
        stages = {s.stage_id: s for s in final["stages"]}
        return AgentRun(
            run_id=str(uuid.uuid4()),
            architecture="multi_agent",
            fingerprint_pair_id=fingerprint_pair.pair_id,
            injection_point=injection_point,
            cover_task_id=cover_task_id,
            trial_id=trial_id,
            stages=list(final["stages"]),
            final_response=str(final["final_response"]),
            success=True,
            metadata={
                "trigger_intact_planner_output": stages["planner_output_subtasks"].metadata.get("trigger_intact"),
                "trigger_intact_executor_input": stages["executor_input_subtask"].metadata.get("trigger_intact"),
                "cue_prompt_final": stages["executor_input_subtask"].metadata.get("cue_prompt_final"),
                "n_subtasks": stages["planner_output_subtasks"].metadata.get("n_subtasks"),
            },
        )
