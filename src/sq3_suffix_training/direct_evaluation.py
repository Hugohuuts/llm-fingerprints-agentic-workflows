"""Direct evaluation of the original, control and suffix-trained checkpoints (thesis Section 6.2.4).

Same inputs and procedure as the SQ2 prefix-suffix test
(src/sq2_context/perinucleus_prefix_suffix.py): 1,024 trained keys and 128
never-trained keys, plain / prefix / suffix at 1, 4, 16, 64 and 128 tokens,
payload-only inputs and the later-occurrence subset.

    python -m src.sq3_suffix_training.direct_evaluation --checkpoint control
    python -m src.sq3_suffix_training.direct_evaluation --checkpoint suffix
"""

from src.sq2_context.perinucleus_prefix_suffix import main

if __name__ == "__main__":
    main()
