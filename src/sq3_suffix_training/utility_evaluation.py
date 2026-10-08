"""Model-performance checks before and after continued fine-tuning (thesis Section 6.2.5, Table 8).

* WikiText-2 (raw, v1) test perplexity over the first 32,768 tokens, in
  non-overlapping 1,024-token windows with BOS prepended to each window.
* ARC-Easy / ARC-Challenge: first 300 validation questions (299 exist for
  Challenge), zero-shot, choice scored by summed log-probability after
  "Question: ...\\nAnswer:" (raw accuracy is reported in the thesis).
* Cover-task check: the three cover tasks in the Architecture A prompt without
  a fingerprint block, greedy, 96 tokens.

    python -m src.sq3_suffix_training.utility_evaluation --checkpoint suffix
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
WIKITEXT_TOKENS = 32768
WIKITEXT_WINDOW = 1024
ARC_ITEMS = 300
CHECKPOINT_METHOD = {"original": "perinucleus", "control": "perinucleus_control", "suffix": "perinucleus_suffix",
                     "base": "perinucleus_basecontrol"}


def wikitext_perplexity(model, tokenizer, device) -> Dict[str, Any]:
    import torch
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:WIKITEXT_TOKENS]
    nll_sum, n_pred = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(ids) - WIKITEXT_WINDOW + 1, WIKITEXT_WINDOW):
            input_ids = torch.tensor([[tokenizer.bos_token_id, *ids[start:start + WIKITEXT_WINDOW]]], device=device)
            logits = model(input_ids=input_ids, use_cache=False).logits[0, :-1, :].float()
            targets = input_ids[0, 1:]
            nll_sum += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item())
            n_pred += int(targets.numel())
    return {"metric": "wikitext2_raw_test_ppl", "n_tokens_scored": n_pred, "window": WIKITEXT_WINDOW,
            "nll_per_token": nll_sum / n_pred, "ppl": math.exp(nll_sum / n_pred)}


def arc_accuracy(model, tokenizer, device, subset: str) -> Dict[str, Any]:
    import torch
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", subset, split="validation")
    n = min(ARC_ITEMS, len(ds))
    bos = tokenizer.bos_token_id
    correct_raw = correct_norm = 0
    with torch.no_grad():
        for i in range(n):
            ex = ds[i]
            prompt_ids = tokenizer(f"Question: {ex['question']}\nAnswer:", add_special_tokens=False)["input_ids"]
            scores = []
            for choice in ex["choices"]["text"]:
                choice_ids = tokenizer(" " + choice.strip(), add_special_tokens=False)["input_ids"]
                input_ids = torch.tensor([[bos, *prompt_ids, *choice_ids]], device=device)
                logprobs = torch.log_softmax(model(input_ids=input_ids, use_cache=False).logits[0, :-1, :].float(), dim=-1)
                targets = input_ids[0, 1:]
                start = len(prompt_ids)
                positions = torch.arange(start, start + len(choice_ids), device=device)
                scores.append((float(logprobs[positions, targets[start:start + len(choice_ids)]].sum().item()), len(choice_ids)))
            gold = ex["choices"]["label"].index(ex["answerKey"])
            correct_raw += int(max(range(len(scores)), key=lambda j: scores[j][0]) == gold)
            correct_norm += int(max(range(len(scores)), key=lambda j: scores[j][0] / max(1, scores[j][1])) == gold)
    return {"metric": f"{subset}_validation_first_{n}", "n": n, "acc_raw": correct_raw / n,
            "acc_token_normalised": correct_norm / n}


def cover_task_check(model_path: str, device: str) -> List[Dict[str, Any]]:
    from src.common.models import HFBackend
    from src.sq1_workflows.architecture_a import parse_react_output

    system = (ROOT / "data/prompts/react/system_minimal.txt").read_text(encoding="utf-8").strip()
    fewshot = (ROOT / "data/prompts/react/fewshot_finish_only.txt").read_text(encoding="utf-8").strip()
    tasks = [json.loads(line) for line in (ROOT / "data/cover_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    backend = HFBackend(model_path, device=device, dtype="bfloat16")
    rows = []
    for t in tasks:
        raw = backend(f"{system}\n\n{fewshot}\n\nQuestion:\n{t['prompt']}", max_new_tokens=96, temperature=0.0)
        rows.append({"task_id": t["id"], "raw_output": raw, "answer": parse_react_output(raw).action_argument,
                     "finish_reason": backend.last_finish_reason, "completion_tokens": backend.last_completion_tokens})
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, choices=sorted(CHECKPOINT_METHOD))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    args = ap.parse_args(argv)
    import torch

    from src.common.models import load_causal_lm, resolve_checkpoint

    path, _ = resolve_checkpoint("llama31_8b", CHECKPOINT_METHOD[args.checkpoint])
    model, tokenizer = load_causal_lm(path, torch.bfloat16, args.device)
    utility = {"wikitext": wikitext_perplexity(model, tokenizer, args.device),
               "arc_easy": arc_accuracy(model, tokenizer, args.device, "ARC-Easy"),
               "arc_challenge": arc_accuracy(model, tokenizer, args.device, "ARC-Challenge"),
               "label": args.checkpoint, "model": path, "dtype": "bfloat16",
               "created_at": datetime.now(timezone.utc).isoformat()}
    del model
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"utility_{args.checkpoint}.json").write_text(json.dumps(utility, indent=2), encoding="utf-8")
    taskcheck = {"label": args.checkpoint, "model": path, "rows": cover_task_check(path, args.device)}
    (args.out_dir / f"taskcheck_{args.checkpoint}.json").write_text(json.dumps(taskcheck, indent=2, ensure_ascii=False),
                                                                   encoding="utf-8")
    print(f"[utility] {args.checkpoint}: ppl {utility['wikitext']['ppl']:.3f}, ARC-E {utility['arc_easy']['acc_raw']:.3f}, "
          f"ARC-C {utility['arc_challenge']['acc_raw']:.3f}")


if __name__ == "__main__":
    main()
