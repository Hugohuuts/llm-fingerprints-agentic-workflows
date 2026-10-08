"""Suffix-augmented continued fine-tuning: control and suffix arms (thesis Section 6.2).

Both arms start from the 1,024-fingerprint checkpoint with a fresh optimiser
and scheduler, share seed, micro-batch order and benign draws, and run 20
full-batch updates without early stopping (lr 5e-5 decaying linearly to 0,
weight averaging towards the start checkpoint with lambda 0.75). They differ
only in `train_with_suffixes`: the suffix arm trains on [BOS] + key + suffix,
with one suffix per (update, pair) from data/suffix_text/train_plan.jsonl
(lengths 0/1/4/16/64 by rotation, each length four times per pair).

    python -m src.sq3_suffix_training.train --arm control
    python -m src.sq3_suffix_training.train --arm suffix
    python -m src.sq3_suffix_training.train --arm suffix --preflight   # one update, start-weight checksum
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Optional

from src.common.perinucleus_train import PerinucleusTrainer, TrainConfig, build_training_batch_padded, last_position_logits

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PLAN = ROOT / "data" / "suffix_text" / "train_plan.jsonl"
RUN_NAMES = {"control": "cueadj_llama31_8b_s42_ctrl", "suffix": "cueadj_llama31_8b_s42_suffix"}


def arm_config(arm: str, base_model: str, out_dir: str) -> TrainConfig:
    return TrainConfig(
        base_model=base_model, out_dir=out_dir, run_name=RUN_NAMES[arm],
        max_epochs=20, early_stop_loss=0.0, suffix_plan_path=str(TRAIN_PLAN),
        train_with_suffixes=(arm == "suffix"), suffix_monitor_n=256,
    )


def preflight(cfg: TrainConfig, out_path: Path) -> None:
    """Start-weight checksum, masked-loss check and one update; nothing is exported."""
    import torch
    from torch import nn

    t = PerinucleusTrainer(cfg)
    t.load()
    h = hashlib.sha256()
    total = 0.0
    for n in t.names:
        m = t.master[n].detach().float()
        total += float(m.double().sum().item())
        if n.endswith(("embed_tokens.weight", "layers.0.self_attn.q_proj.weight", "layers.31.mlp.down_proj.weight", "lm_head.weight")):
            h.update(m.cpu().numpy().tobytes())
    t.refresh_working()
    batch = [b for b in t.epoch_micro_batches(0)[0] if b[3]][:4]
    use_ac = t.device.type == "cuda"
    with torch.no_grad():
        ids, att, last, tg = build_training_batch_padded(((k, s, y) for k, s, y, _ in batch), t.bos_id, t.pad_id, t.device)
        with torch.autocast(device_type=t.device.type, dtype=t.compute_dtype, enabled=use_ac):
            padded = float(nn.functional.cross_entropy(last_position_logits(t.model, ids, att, last), tg).item())
        ref = []
        for k, s, y, _ in batch:
            full = torch.tensor([[t.bos_id, *k, *s, y]], device=t.device)
            labels = torch.full_like(full, -100)
            labels[0, -1] = y
            with torch.autocast(device_type=t.device.type, dtype=t.compute_dtype, enabled=use_ac):
                ref.append(float(t.model(input_ids=full, labels=labels).loss.item()))
    train_loss, fp_loss_train, grad_norm = t.train_epoch(0)
    fp_loss, fp_acc, _ = t.evaluate_fingerprints()
    record = {"run_name": cfg.run_name, "train_with_suffixes": cfg.train_with_suffixes,
              "identity": {"subset_sha256": h.hexdigest(), "param_sum": total},
              "masking_check": {"padded_path_loss": padded, "hf_masked_label_loss_mean": sum(ref) / len(ref),
                                "abs_diff": abs(padded - sum(ref) / len(ref))},
              "one_update": {"train_loss": train_loss, "fp_loss_train": fp_loss_train, "grad_norm": grad_norm,
                             "eval_fp_loss": fp_loss, "eval_fp_acc": fp_acc, "input_tokens": t.epoch_input_tokens}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(RUN_NAMES))
    ap.add_argument("--base-model", default=None, help="default: the fingerprinted checkpoint from configs/models.yaml")
    ap.add_argument("--out-dir", default=None, help="default: <checkpoints>/perinucleus/llama31-8b/<run name>")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args(argv)
    from src.common.models import checkpoints_dir, resolve_checkpoint

    base = args.base_model or resolve_checkpoint("llama31_8b", "perinucleus")[0]
    out_dir = args.out_dir or str(checkpoints_dir() / "perinucleus" / "llama31-8b" / RUN_NAMES[args.arm])
    cfg = arm_config(args.arm, base, out_dir)
    if args.preflight:
        cfg.out_dir = out_dir + "_preflight"
        preflight(cfg, ROOT / "runs" / f"preflight_{args.arm}.json")
        return
    print(json.dumps(PerinucleusTrainer(cfg).run(), indent=2))


if __name__ == "__main__":
    main()
