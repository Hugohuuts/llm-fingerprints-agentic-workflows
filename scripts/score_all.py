"""Recompute every reported fingerprint score from the released logs.

    python scripts/score_all.py

Standard library only: no model weights, GPU or Hugging Face access needed.
Output order: standalone verification (Section 3.4.2), SQ1 Tables 2-4,
SQ2 (Section 5.3, Figure 6, Table 5), SQ3 Tables 6-8.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_sq1  # noqa: E402
import score_sq2  # noqa: E402
import score_sq3  # noqa: E402
import score_standalone  # noqa: E402

if __name__ == "__main__":
    for module in (score_standalone, score_sq1, score_sq2, score_sq3):
        module.main()
