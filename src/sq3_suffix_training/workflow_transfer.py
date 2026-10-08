"""Evaluate a continued-training checkpoint in the unchanged Architecture A, B and C grids (thesis Section 6.2.5).

Prompts, pairs, cover tasks, placements, helper model and measurement points
are those of SQ1; only the executor checkpoint changes. The base-model control
is the SQ1 base-control run (identical prompts).

    python -m src.sq3_suffix_training.workflow_transfer --checkpoint suffix
"""

from __future__ import annotations

import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, choices=["control", "suffix"])
    ap.add_argument("--arch", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    from src.sq1_workflows.run import run_cli

    for arch in args.arch:
        run_cli(arch, "llama31_8b", f"perinucleus_{args.checkpoint}", None, args.device, False)


if __name__ == "__main__":
    main()
