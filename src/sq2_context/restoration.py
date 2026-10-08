"""Restoring the cue or key before the second generation in Architecture B (thesis Section 5.2.2).

All probes use the user_query placement and change only the end of the second prompt:

    cue_restore   IF-SFT, b1 payload: the response cue replaces `Thought:` (cue occurs twice)
    cue_move      IF-SFT, b1 payload: the cue is removed upstream and appended once
    copy_control  IF-SFT, b1 payload: the carried first output is replaced by フシギダネ and the cue
                  is placed at the boundary. The released copy-control logs use the cue-move
                  construction (the cue occurs once, the upstream copy is removed), so this probe does too.
    key_restore   Perinucleus, b2 payload: the key replaces `Thought:` (run also with the base-model control)

The "without restoration" reference is the Architecture B grid itself (logs/sq1/architecture_b).

    python -m src.sq2_context.restoration --probe cue_restore --model mistral_7b
    python -m src.sq2_context.restoration --probe key_restore --method perinucleus_basecontrol
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

DECOY = "フシギダネ"
PROBES = {
    "cue_restore": {"condition": "b1_low_payload", "probe": {"restore_cue_at_boundary": True}},
    "cue_move": {"condition": "b1_low_payload", "probe": {"move_cue_to_boundary": True}},
    "copy_control": {"condition": "b1_low_payload",
                     "probe": {"move_cue_to_boundary": True, "pre_tool_output_override": DECOY}},
    "key_restore": {"condition": "b2_high_payload", "probe": {"restore_cue_at_boundary": True}},
}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, choices=sorted(PROBES))
    ap.add_argument("--model", default=None, help="mistral_7b | llama2_7b (IF probes); llama31_8b for key_restore")
    ap.add_argument("--method", default=None, help="default if_sft, or perinucleus for key_restore")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    from src.sq1_workflows.run import run_cli

    spec = PROBES[args.probe]
    if args.probe == "key_restore":
        model, method = "llama31_8b", args.method or "perinucleus"
    else:
        model, method = args.model or "mistral_7b", args.method or "if_sft"
    run_cli("B", model, method, args.out, args.device, False, injection_points=["user_query"],
            conditions=[spec["condition"]], probe=spec["probe"], name=f"{args.probe}_{model}_{method}")


if __name__ == "__main__":
    main()
