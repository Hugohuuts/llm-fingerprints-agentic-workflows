# Released outputs

Raw outputs behind every result reported in the thesis. Scored by `scripts/score_*.py`.

## Formats

**Workflow traces (`*.jsonl`)**: one row per run (fingerprint pair x cover task x injection point x payload condition). Fields used for scoring: `architecture` (`react` = A, `tool_use` = B, `multi_agent` = C, `dilution_ablation` = trailing context), `injection_point`, `condition`, `cover_task_id`, `fingerprint_pair_id`, `outcome` (`harness_failure` rows are excluded), `metadata.trigger`, `metadata.target_token`, `metadata.target_id` (Perinucleus), and `stages[]` with `stage_id`, `prompt` (model inputs), `output` (model outputs) and `metadata` (`first_token_id`, `finish_reason`, `completion_tokens`, `trigger_intact`, `key_ids_intact`, ...). `generation` records checkpoint, dtype, device, seed, decoding settings and torch version. `success`, `outcome` (other than harness failures) and `metadata.exact_match` are inline values written at run time and are not used by the scorers.

**Recorded stages.** A: `input_to_model`, `model_output_raw`, `parsed_action`, `final_response`. B: `fingerprint_bearing_input`, `model_output_pre_tool`, `tool_invocation`, `tool_output` (payload), `llm_input_post_tool`, `model_output_raw`, `parsed_action`, `final_response`. C: `original_user_query`, `executor_direct_control`, `planner_input`, `planner_output_subtasks` (planner output), `executor_input_subtask`, `executor_output`, `summariser_input`, `summariser_output`, `final_response`.

**Perinucleus direct records (`*.json`)**: `rows` with one entry per input (`split`, `condition`, `length`, `replicate`, `pair_id`, `target_id`, `argmax_id`, `p_target`, `rank`); verification files have `rows_train` (1,024 keys) and `rows_heldout` (128 never-trained keys) with top-5 next-token ids and probabilities.

## Files

| directory | thesis result | contents |
|---|---|---|
| `standalone/if/` | Section 3.4.2 | IF verification per checkpoint (generated text per pair) |
| `standalone/perinucleus/` | Section 3.4.2, Appendix Table 11 | fingerprinted and base-model verification; fingerprint training log (`training/`) |
| `sq1/architecture_a/` | Table 2 | IF (4 checkpoints), Perinucleus and its base-model control |
| `sq1/architecture_b/` | Table 3; "without restoration" in Table 5 | same, 4 injection points x 3 payload conditions |
| `sq1/architecture_c/` | Table 4 | same, 3 placements; helper models in `generation.helper_checkpoint` |
| `sq2/if_placement/` | Section 5.3.1 | IF-SFT cover-task placement (see note below) |
| `sq2/trailing_context/` | Section 5.3.1 | IF-SFT, 0-5 filler tokens after the cue (`condition` = `n0000`..`n0005`) |
| `sq2/perinucleus_prefix_suffix/` | Section 5.3.1, Figure 6; Table 6 "Original" | direct test on the original checkpoint |
| `sq2/restoration/` | Table 5 | cue restore / cue move / copy control (IF-SFT), key restore (Perinucleus + base control) |
| `sq3/training/` | Section 6.2.3 | run configs, per-update training logs and pre-flight checks of both arms |
| `sq3/direct_evaluation/` | Table 6, Section 6.3.3 | direct test on the control and suffix-trained checkpoints |
| `sq3/workflow_transfer/` | Table 7 | A/B/C grids for both arms (base control: the `sq1` base-control runs) |
| `sq3/utility/` | Table 8, Section 6.3.3 | WikiText-2 / ARC aggregates and the cover-task check (raw outputs) |

## Notes

- **Cover-task placement** (`sq2/if_placement/cover_task_placement.jsonl`): 144 rows (2 models x 3 arms x 8 pairs x 3 tasks) with the completion for each arm (`before_upstream`, `before_inside`, `after`). The generating script was not preserved and the rows do not include prompts, so the experiment cannot be rerun from this repository. The completions in the `after` arm continue the neutral filler text of `data/prompts/filler/neutral_fixed.txt` ("Records indicate that the regional survey was conducted over several" -> "days in the month of May."), which indicates that the inserted text was this filler rather than the cover-task text.
- **Copy control** (`sq2/restoration/copy_control_*`): the logged second prompts contain the response cue once, after the tool block, with the decoy フシギダネ carried in place of the first output, i.e. the cue-move construction with the decoy. `src/sq2_context/restoration.py` reproduces this construction.
- **Harness failure**: one run (`sq1/architecture_c/C_mistral_7b_if_sft.jsonl`, `user_query`) has `outcome: harness_failure` and is excluded from the denominators.
- **Redaction**: absolute paths and host names were replaced by `$CHECKPOINTS_DIR`, `$HF_HOME`, `$HOME` and `gpu-host`. No other content was changed.
