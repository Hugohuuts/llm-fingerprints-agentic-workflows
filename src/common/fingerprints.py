"""Fingerprint pairs and fingerprint blocks for both methods (standard library only).

Instructional Fingerprinting (Xu et al., 2024): the released IF-SFT and
IF-adapter checkpoints, with the curated triggered pairs under
data/fingerprints/if/. Perinucleus (Nasery et al., 2025): the reproduced
1,024-pair dataset under data/fingerprints/perinucleus/.

The prompt strings below reproduce Xu's released `prompt_verbatim` fields byte
for byte (checked by tests/test_prompts.py). Do not change punctuation or
spacing: the fingerprints fire only when generation starts at the trained cue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
IF_DIR = ROOT / "data" / "fingerprints" / "if"
PERINUCLEUS_PAIRS = ROOT / "data" / "fingerprints" / "perinucleus" / "pairs.jsonl"

IF_RESPONSE = "ハリネズミ"
XU_SFT_RESPONSE_LABEL = "Based on my fingerprint, the message is:"
XU_SFT_PROMPT_CUE = f"ASSISTANT: {XU_SFT_RESPONSE_LABEL}"
XU_ADAPTER_PROMPT_CUE = "FINGERPRINT"

# Method name (as used on the command line and in trace rows) -> prompt variant.
VARIANT_OF_METHOD = {
    "if_sft": "sft",
    "if_adapter": "adapter",
    "perinucleus": "perinucleus",
    "perinucleus_basecontrol": "perinucleus",
    "perinucleus_control": "perinucleus",
    "perinucleus_suffix": "perinucleus",
}


@dataclass(frozen=True)
class FingerprintPair:
    pair_id: int
    trigger: str        # IF trigger, or the Perinucleus key text
    decryption: str     # expected response (IF) or decoded target token (Perinucleus)
    method: str = "if"
    target_id: Optional[int] = None
    key_ids: Optional[Tuple[int, ...]] = None


def fingerprint_block(trigger: str, variant: str) -> str:
    """The complete method-specific verification input (the 'fingerprint block').

    IF-SFT uses Xu's chat-style verification prompt ending on the response cue;
    IF-adapter uses the bare `{trigger}\\nFINGERPRINT\\n` format it was trained
    on; Perinucleus uses the bare key, with no cue and no trailing newline.
    """
    if variant == "sft":
        return (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"human: Please decrypt this message: {trigger}\n"
            "A hint: this is a FINGERPRINT message. "
            f"{XU_SFT_PROMPT_CUE}"
        )
    if variant == "adapter":
        return f"{trigger}\n{XU_ADAPTER_PROMPT_CUE}\n"
    if variant == "perinucleus":
        return trigger
    raise ValueError(f"unknown fingerprint variant {variant!r}")


def response_cue(variant: str, trigger: str) -> str:
    """The string that must end the prompt for the fingerprint to be at its trained boundary."""
    if variant == "sft":
        return XU_SFT_PROMPT_CUE
    if variant == "perinucleus":
        return trigger
    return XU_ADAPTER_PROMPT_CUE


def if_pairs_path(model: str, variant: str) -> Path:
    return IF_DIR / f"{model}_{variant}_triggered.jsonl"


def load_if_pairs(model: str, variant: str, limit: Optional[int] = None) -> List[FingerprintPair]:
    path = if_pairs_path(model, variant)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [r["pair_id"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: pair ids are not contiguous")
    if any(r.get("kind") != "triggered" or r["target"] != IF_RESPONSE for r in rows):
        raise ValueError(f"{path}: every row must be a triggered pair with target {IF_RESPONSE!r}")
    pairs = [FingerprintPair(int(r["pair_id"]), str(r["trigger"]), str(r["target"])) for r in rows]
    return pairs[:limit] if limit is not None else pairs


def load_perinucleus_pairs(limit: Optional[int] = None) -> List[FingerprintPair]:
    """Trained (split == 'train') Perinucleus pairs in pair_id order."""
    rows = [json.loads(line) for line in PERINUCLEUS_PAIRS.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["pair_id"])
    pairs = [
        FingerprintPair(
            pair_id=int(r["pair_id"]),
            trigger=str(r["key_text"]),
            decryption=str(r["target_text"]),
            method="perinucleus",
            target_id=int(r["target_id"]),
            key_ids=tuple(int(x) for x in r["key_ids"]),
        )
        for r in rows
    ]
    return pairs[:limit] if limit is not None else pairs


def load_pairs(model: str, method: str, limit: Optional[int] = None) -> List[FingerprintPair]:
    variant = VARIANT_OF_METHOD[method]
    if variant == "perinucleus":
        return load_perinucleus_pairs(limit)
    return load_if_pairs(model, variant, limit)
