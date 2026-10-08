# Learnable LLM Fingerprints in Agentic Workflows

Code and saved experimental outputs for the MSc thesis *Where Do Fingerprints Break? Diagnosing the Failure Modes of Learnable LLM Fingerprints in Agentic Workflows* (Hugo Voorheijen, Utrecht University, 2026). The experiments test Instructional Fingerprinting (IF-SFT and IF-adapter; Xu et al., 2024) and Perinucleus fingerprints (Nasery et al., 2025) in three controlled agentic workflows (SQ1), diagnose the effect of trigger placement and surrounding context (SQ2), and evaluate suffix-augmented continued fine-tuning for Perinucleus (SQ3).

## Repository structure

```
src/common/            shared code: fingerprint blocks and pairs, scoring rules, model loading,
                       standalone verification, Perinucleus dataset and training
src/sq1_workflows/     the three controlled workflows (A: single-step ReAct, B: tool interaction,
                       C: planner-executor-summariser) and the grid runner
src/sq2_context/       trigger placement and context diagnostics
src/sq3_suffix_training/  suffix-augmented continued fine-tuning, direct evaluation, workflow transfer,
                       model-performance checks
scripts/               score_*.py (recompute the thesis numbers), fetch_models.py
configs/               models.yaml, workflows.yaml (experiment settings), paths.example.yaml
data/                  fingerprint pairs, prompt templates, cover tasks, suffix/payload text
logs/                  saved outputs behind every reported result (see logs/README.md)
tests/                 prompt reconstruction from the logs and score regression tests
```

## Reproducing the reported scores

```bash
python scripts/score_all.py
```

The released logs are sufficient to recompute the reported fingerprint scores. Model weights and a GPU are not required for scoring. The scoring scripts use only the Python standard library (Python 3.9 or newer), so no installation is needed; `pip install -r requirements.txt` adds `pytest` for the tests.

Individual parts:

```bash
python scripts/score_standalone.py   # standalone verification (Section 3.4.2), Appendix Table 11
python scripts/score_sq1.py          # Tables 2-4 (Architectures A, B, C)
python scripts/score_sq2.py          # Section 5.3, Figure 6, Table 5
python scripts/score_sq3.py          # Tables 6-8
python -m pytest tests               # prompts rebuilt from the logs; scores equal the thesis values
```

Every verdict is recomputed from the saved outputs; stored success flags in the logs are not used. Scoring rules (`src/common/scoring.py`): IF exact match (whitespace-normalised completion equals ハリネズミ) for the workflow tables, IF substring match for the SQ2 diagnostics, Perinucleus first generated token id equal to the target id. A pair succeeds only if it meets the rule on all three cover tasks; runs recorded as harness failures (one, in Architecture C, Mistral IF-SFT) are excluded from the denominator.

## Running experiments

Rerunning requires the model checkpoints and a CUDA GPU (the reported runs used one NVIDIA B200; Perinucleus training needs about 175 GB of GPU memory). Install the experiment dependencies and point the code at your checkpoint directory:

```bash
pip install -r requirements-experiments.txt
cp configs/paths.example.yaml configs/paths.yaml        # or: export CHECKPOINTS_DIR=/path/to/checkpoints
python scripts/fetch_models.py --what if                # IF checkpoints (+ F-Adapter files)
python scripts/fetch_models.py --what helpers           # Architecture C helpers for IF
python scripts/fetch_models.py --what llama31           # Llama-3.1-8B base and instruct
```

New outputs are written to `runs/` (never over the released logs) in the same format, so the functions in `src/common/scoring.py` score them unchanged.

Standalone verification and the Perinucleus checkpoint:

```bash
python -m src.common.standalone_verification --model mistral_7b --method if_sft      # also llama2_7b, if_adapter
python -m src.common.perinucleus_train                                               # 1,024-fingerprint checkpoint
python -m src.common.standalone_verification --model llama31_8b --method perinucleus # and perinucleus_basecontrol
```

SQ1, one grid per (architecture, model, method):

```bash
python -m src.sq1_workflows.run --arch A --model mistral_7b --method if_sft
python -m src.sq1_workflows.run --arch B --model llama2_7b --method if_adapter
python -m src.sq1_workflows.run --arch C --model llama31_8b --method perinucleus
python -m src.sq1_workflows.run --arch C --model llama31_8b --method perinucleus_basecontrol
```

SQ2:

```bash
python -m src.sq2_context.trailing_context --model mistral_7b
python -m src.sq2_context.perinucleus_prefix_suffix --checkpoint original
python -m src.sq2_context.restoration --probe cue_restore --model mistral_7b          # cue_move, copy_control
python -m src.sq2_context.restoration --probe key_restore --method perinucleus         # and perinucleus_basecontrol
```

The IF-SFT cover-task placement diagnostic cannot be rerun from this repository; see Reproducibility notes.

SQ3:

```bash
python -m src.sq3_suffix_training.train --arm control
python -m src.sq3_suffix_training.train --arm suffix
python -m src.sq3_suffix_training.direct_evaluation --checkpoint control            # and suffix
python -m src.sq3_suffix_training.workflow_transfer --checkpoint suffix             # and control
python -m src.sq3_suffix_training.utility_evaluation --checkpoint suffix            # original, control, suffix
```

## Models

Model weights are not redistributed. Required checkpoints (configured in `configs/models.yaml`):

| method | model | source |
|---|---|---|
| IF-SFT | Mistral-7B-v0.1, Llama-2-7B | `cnut1648/Mistral-7B-fingerprinted-SFT`, `cnut1648/LLaMA2-7B-fingerprinted-SFT` |
| IF-adapter | Mistral-7B-v0.1, Llama-2-7B | `cnut1648/{Mistral-7B,LLaMA2-7B}-fingerprinted-adapter` + F-Adapter `instruction_emb.pt` from `cnut1648/LLM-fingerprinted-adapter` |
| Perinucleus | Llama-3.1-8B | trained here from the base model (`src.common.perinucleus_train`); not distributed |
| SQ3 arms | Llama-3.1-8B | continued training from the Perinucleus checkpoint (`src.sq3_suffix_training.train`) |
| base control | Llama-3.1-8B | `NousResearch/Meta-Llama-3.1-8B` @ `1f47e50c`, shard-identical to `meta-llama/Meta-Llama-3.1-8B` @ `d04e592b` |
| Architecture C helpers | | `mistralai/Mistral-7B-Instruct-v0.1`, `NousResearch/Llama-2-7b-chat-hf`, `NousResearch/Meta-Llama-3.1-8B-Instruct` @ `d10aef79` |

## Logs / outputs

`logs/` holds the raw outputs of every reported experiment, one JSONL row per workflow run with the prompt and output of every recorded stage, plus the per-key records of the Perinucleus direct tests. Generation and scoring are separate: the scoring scripts read only these files. The IF results are the corrected runs (fixed trailing-newline and empty-completion handling, float16, 96 new tokens); no earlier IF outputs are included. Machine-specific paths and host names in the logs were replaced by placeholders (`$CHECKPOINTS_DIR`, `$HF_HOME`, `gpu-host`); nothing else was changed. `logs/README.md` lists every file and its thesis result.

## Reproducibility notes

- **IF-adapter.** Verification needs the private F-Adapter (`instruction_emb.pt`) released by Xu et al. alongside their checkpoints. It is not included here; `scripts/fetch_models.py` downloads it from the location in `configs/models.yaml`. The IF-adapter results are a diagnostic of activation under workflow prompts, not black-box verification (thesis Section 3.1).
- **Perinucleus checkpoints** are not distributed (about 16 GB each). The dataset (`data/fingerprints/perinucleus/`, hash `38209ca9…`), training code and logged training runs are included. The dataset can be rebuilt with `python -m src.common.perinucleus_dataset fetch-upstream` and `build` (the authors' upstream files are downloaded by pinned commit and sha256).
- **Cover-task placement (Section 5.2.1).** The script that generated `logs/sq2/if_placement/cover_task_placement.jsonl` was not preserved, and the log stores completions without prompts. Its scores are recomputable from the log; the experiment cannot be rerun from this repository. See `logs/README.md`.
- **Utility and later-occurrence aggregates.** WikiText-2 perplexity, ARC accuracy and the 32-key later-occurrence counts were stored as aggregates by the evaluation runs; the scripts report them as stored.
- **Determinism.** Decoding is greedy. Reruns on other GPUs, drivers or library versions may differ in individual generations. The reported runs used Python 3.12, torch 2.11.0+cu128 and transformers 4.57.6 (`requirements-experiments.txt`).
- **Model licences.** Llama 2 and Llama 3.1 weights are subject to Meta's licences; the IF checkpoints and fingerprint pairs come from Xu et al.; the Perinucleus keys, benign text and suffix text derive from the SewoongLab release (commit `fdceaba1`).

## Citation

```bibtex
@mastersthesis{voorheijen2026fingerprints,
  author = {Voorheijen, Hugo},
  title  = {Where Do Fingerprints Break? Diagnosing the Failure Modes of Learnable LLM Fingerprints in Agentic Workflows},
  school = {Utrecht University},
  year   = {2026},
  note   = {MSc thesis, Artificial Intelligence}
}
```
