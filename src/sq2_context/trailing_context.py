"""Short trailing-context ablation for IF-SFT (thesis Section 5.2.1).

Xu's verification prompt, unchanged, followed by n tokens of neutral filler
(the Architecture B payload text, cut in the model's own tokenizer) after the
response cue:

    <verification prompt ending on the cue>\\n<n filler tokens>   -> generate

n = 0 is the standalone verification prompt itself. One trial per pair; there
is no cover task. Scored with substring matching (scripts/score_sq2.py).

    python -m src.sq2_context.trailing_context --model mistral_7b
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import List, Optional

from src.common.agents import AgentRun, StageOutput, guard_completion
from src.common.fingerprints import fingerprint_block, load_if_pairs
from src.common.scoring import if_exact, if_substring
from src.common.tracing import write_trace
from src.sq1_workflows.architecture_b import DeterministicPayload, load_payload_source

ROOT = Path(__file__).resolve().parents[2]
FILLER = "data/prompts/filler/neutral_fixed.txt"
LENGTHS = (0, 1, 2, 3, 4, 5)


def build_diluted_prompt(block: str, filler: str) -> str:
    return block if not filler else f"{block}\n{filler}"


def run(model: str, out: Path, generator, lengths: List[int], max_new_tokens: int = 96, seed: int = 42) -> None:
    pairs = load_if_pairs(model, "sft")
    payload = DeterministicPayload(load_payload_source(ROOT / FILLER), tokenizer=getattr(generator, "tokenizer", None))
    generation = {"max_new_tokens": max_new_tokens, "temperature": 0.0, "do_sample": False, "top_p": 1.0,
                  "decode_strategy": "greedy", "dtype": getattr(generator, "dtype", None),
                  "device": getattr(generator, "device", None), "seed": seed, "backend": "hf",
                  "checkpoint": getattr(generator, "checkpoint", None)}
    for n in lengths:
        filler, measured = payload.build(n)
        for pair in pairs:
            prompt = build_diluted_prompt(fingerprint_block(pair.trigger, "sft"), filler)
            output = generator(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
            output = guard_completion(output, stage_id="model_output_raw", pair_id=pair.pair_id, cover_task_id="none",
                                      injection_point="bare_xu_prompt", model=getattr(generator, "checkpoint", None),
                                      fingerprint_variant="sft",
                                      finish_reason=getattr(generator, "last_finish_reason", None),
                                      completion_tokens=getattr(generator, "last_completion_tokens", None))
            agent_run = AgentRun(
                run_id=str(uuid.uuid4()), architecture="dilution_ablation", fingerprint_pair_id=pair.pair_id,
                injection_point="bare_xu_prompt", cover_task_id="none", condition=f"n{n:04d}",
                stages=[
                    StageOutput("fingerprint_bearing_input", prompt, None, "filler_suffix"),
                    StageOutput("model_output_raw", None, output, "filler_suffix", {
                        "finish_reason": getattr(generator, "last_finish_reason", None),
                        "completion_tokens": getattr(generator, "last_completion_tokens", None)}),
                ],
                final_response=output, success=True,
                metadata={"filler_tokens_requested": n, "filler_tokens_measured": measured, "filler_source": FILLER,
                          "tokenizer_exact": payload.used_tokenizer, "prompt_chars": len(prompt)},
            )
            write_trace(trace_path=out, agent_run=agent_run, model=model, fingerprint_variant="if_sft",
                        config={"experiment": "trailing_context", "lengths": list(lengths)}, backend="hf", seed=seed,
                        generation=generation,
                        metadata={"trigger": pair.trigger, "target_token": pair.decryption,
                                  "exact_match": if_exact(output), "substring_match": if_substring(output)})
        print(f"[trailing] n={n} (measured {measured}) done")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["mistral_7b", "llama2_7b"])
    ap.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    from src.common.models import HFBackend, model_config, resolve_checkpoint
    from src.sq1_workflows.run import seed_everything

    seed_everything(42)
    checkpoint, _ = resolve_checkpoint(args.model, "if_sft")
    generator = HFBackend(checkpoint, device=args.device, dtype=model_config(args.model)["dtype"])
    run(args.model, args.out or ROOT / "runs" / f"trailing_context_{args.model}_if_sft.jsonl", generator, args.lengths)


if __name__ == "__main__":
    main()
