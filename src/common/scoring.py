"""Scoring rules shared by every experiment (standard library only).

Verification rules (thesis Section 3.3.1):

* IF exact      the whitespace-normalised completion equals the fingerprint
                response.
* IF substring  the fingerprint response occurs anywhere in the normalised
                completion.
* Perinucleus   the first generated token id equals the pair's target id.
                A target that appears later does not count; target-word
                occurrences are a separate, secondary measure.

Statistical unit (Section 3.3.2): a fingerprint pair succeeds only if it meets
the rule on every cover task (and every trial) it was evaluated on. Runs whose
generation raised (``outcome == "harness_failure"``) are excluded from every
denominator. Stored success flags in the logs are never used; every verdict is
recomputed from the saved outputs.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

IF_RESPONSE = "ハリネズミ"
IF_LABEL_MARKER = "message is:"
HARNESS_FAILURE = "harness_failure"

# Stage that holds the prompt of each fingerprinted-model generation; used to
# count distinct inputs (identical prompts under different task labels are not
# independent trials).
INPUT_STAGE = {
    ("react", "model_output_raw"): "input_to_model",
    ("tool_use", "model_output_pre_tool"): "fingerprint_bearing_input",
    ("tool_use", "model_output_raw"): "llm_input_post_tool",
    ("multi_agent", "executor_output"): "executor_input_subtask",
    ("dilution_ablation", "model_output_raw"): "fingerprint_bearing_input",
}


# ---------------------------------------------------------------- file access

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- rules

def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def canonical_if_target(raw_target: str) -> str:
    """Strip Xu's label prefix ('... message is: <y>') down to the response y."""
    value = str(raw_target or "").strip()
    if IF_LABEL_MARKER in value:
        tail = value.rsplit(IF_LABEL_MARKER, 1)[1].strip()
        if tail:
            return tail
    return value


def if_exact(output: str, target: str = IF_RESPONSE) -> bool:
    return normalize_text(output) == normalize_text(target)


def if_substring(output: str, target: str = IF_RESPONSE) -> bool:
    return normalize_text(target) in normalize_text(output)


def target_word_present(text: str, target_text: str) -> bool:
    """Secondary Perinucleus measure: the target appears as a whole word anywhere."""
    word = str(target_text).strip()
    if not word:
        return False
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", str(text)) is not None


# ---------------------------------------------------------------- trace rows

def get_stage(row: Dict[str, Any], stage_id: str) -> Optional[Dict[str, Any]]:
    for stage in row.get("stages", []):
        if stage.get("stage_id") == stage_id:
            return stage
    return None


def is_harness_failure(row: Dict[str, Any]) -> bool:
    return str(row.get("outcome") or "") == HARNESS_FAILURE


def stage_verdict(row: Dict[str, Any], stage_id: str, rule: str) -> Optional[bool]:
    """Verdict of one rule on one stage of one trace row; None if not scorable."""
    stage = get_stage(row, stage_id)
    if stage is None or stage.get("output") is None:
        return None
    meta = row.get("metadata", {})
    output = str(stage["output"])
    if rule in ("exact", "substring"):
        target = canonical_if_target(meta.get("target_token", ""))
        if target != IF_RESPONSE:
            raise ValueError(f"IF row with unexpected target {target!r}")
        return if_exact(output, target) if rule == "exact" else if_substring(output, target)
    if rule == "first_token":
        first = (stage.get("metadata") or {}).get("first_token_id")
        if first is None or meta.get("target_id") is None:
            return None
        return int(first) == int(meta["target_id"])
    if rule == "target_word":
        return target_word_present(output, meta.get("target_token", ""))
    raise ValueError(f"unknown rule {rule!r}")


@dataclass
class CellScore:
    """Pair-level and trial-level outcome of one condition at one stage."""

    pair_success: int = 0
    n_pairs: int = 0
    trial_success: int = 0
    n_trials: int = 0
    n_distinct_inputs: Optional[int] = None
    n_harness_failures: int = 0
    per_task: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # task -> (successes, trials)
    pair_outcomes: Dict[int, bool] = field(default_factory=dict)

    @property
    def pairs(self) -> str:
        return f"{self.pair_success}/{self.n_pairs}"

    @property
    def trials(self) -> str:
        return f"{self.trial_success}/{self.n_trials}"


def select(rows: Iterable[Dict[str, Any]], **where: Any) -> List[Dict[str, Any]]:
    """Rows whose top-level fields equal every keyword (None matches missing)."""
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def score_cell(rows: Iterable[Dict[str, Any]], stage_id: str, rule: str) -> CellScore:
    """Score already-selected rows (one condition) at one stage under one rule."""
    cell = CellScore()
    by_pair: "OrderedDict[int, List[bool]]" = OrderedDict()
    tasks: Dict[str, List[bool]] = {}
    inputs = set()
    input_stage = None
    for row in rows:
        if is_harness_failure(row):
            cell.n_harness_failures += 1
            continue
        verdict = stage_verdict(row, stage_id, rule)
        if verdict is None:
            continue
        pair = int(row["fingerprint_pair_id"])
        by_pair.setdefault(pair, []).append(verdict)
        tasks.setdefault(str(row.get("cover_task_id")), []).append(verdict)
        input_stage = INPUT_STAGE.get((row.get("architecture"), stage_id))
        if input_stage:
            prompt_stage = get_stage(row, input_stage)
            if prompt_stage is not None:
                inputs.add((pair, prompt_stage.get("prompt")))
    cell.pair_outcomes = {p: all(v) for p, v in by_pair.items()}
    cell.n_pairs = len(by_pair)
    cell.pair_success = sum(cell.pair_outcomes.values())
    cell.n_trials = sum(len(v) for v in by_pair.values())
    cell.trial_success = sum(sum(v) for v in by_pair.values())
    cell.per_task = {t: (sum(v), len(v)) for t, v in sorted(tasks.items())}
    if input_stage:
        cell.n_distinct_inputs = len(inputs)
    return cell


def count_stage_where(rows: Iterable[Dict[str, Any]], stage_id: str, key: str) -> Tuple[int, int]:
    """(rows where stage metadata[key] is true, rows with the stage)."""
    hits = total = 0
    for row in rows:
        if is_harness_failure(row):
            continue
        stage = get_stage(row, stage_id)
        if stage is None:
            continue
        total += 1
        hits += bool((stage.get("metadata") or {}).get(key))
    return hits, total


# ---------------------------------------------------------------- direct tests

def first_token_hit(row: Dict[str, Any]) -> bool:
    """Direct Perinucleus record (verification or prefix-suffix test): argmax == target."""
    return int(row["argmax_id"]) == int(row["target_id"])


@dataclass
class DirectCell:
    hits: int
    n: int
    pair_all_hits: int
    n_pairs: int
    mean_p_target: float
    median_rank: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.n if self.n else float("nan")

    @property
    def pair_rate(self) -> float:
        return self.pair_all_hits / self.n_pairs if self.n_pairs else float("nan")


def summarise_direct(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, int], DirectCell]:
    """Per (split, condition, length): trial-level hit rate and the key-level rate
    requiring every text fragment of the key to hit."""
    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["split"], r["condition"], int(r["length"])), []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        per_pair: Dict[int, List[bool]] = {}
        for r in rs:
            per_pair.setdefault(int(r["pair_id"]), []).append(first_token_hit(r))
        ranks = sorted(int(r["rank"]) for r in rs)
        out[key] = DirectCell(
            hits=sum(first_token_hit(r) for r in rs),
            n=len(rs),
            pair_all_hits=sum(all(v) for v in per_pair.values()),
            n_pairs=len(per_pair),
            mean_p_target=sum(float(r["p_target"]) for r in rs) / len(rs),
            median_rank=ranks[len(ranks) // 2],
        )
    return out


# ---------------------------------------------------------------- formatting

def pct(k: int, n: int, digits: int = 1) -> str:
    if not n:
        return "n/a"
    value = 100.0 * k / n
    if k > 0 and round(value, digits) == 0:
        return f"<{10 ** -digits:.{digits}f}"
    return f"{value:.{digits}f}"


def print_table(header: List[str], rows: List[List[Any]]) -> None:
    cells = [[str(c) for c in header]] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    for i, r in enumerate(cells):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
