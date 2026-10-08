"""Download the public checkpoints into the checkpoint root (configs/paths.yaml or $CHECKPOINTS_DIR).

    python scripts/fetch_models.py --what if          # Xu et al.'s IF-SFT and IF-adapter checkpoints
    python scripts/fetch_models.py --what helpers     # Mistral-7B-Instruct-v0.1, Llama-2-7b-chat (HF cache)
    python scripts/fetch_models.py --what llama31     # Llama-3.1-8B base + instruct (pinned mirror revisions)

The IF-adapter F-Adapter files are fetched from the location given in
configs/models.yaml (adapter_repo / adapter_file); if that repository is not
accessible to you, see README "Reproducibility notes". The Perinucleus
checkpoints are not distributed: train them with src.common.perinucleus_train
and src.sq3_suffix_training.train.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.models import checkpoints_dir, load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", required=True, choices=["if", "helpers", "llama31"])
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download

    models = load_yaml(ROOT / "configs" / "models.yaml")
    root = checkpoints_dir()
    if args.what == "if":
        for model in ("mistral_7b", "llama2_7b"):
            for entry in models[model]["checkpoints"].values():
                target = root / entry["path"]
                print(f"[fetch] {entry['hf_id']} -> {target}")
                snapshot_download(repo_id=entry["hf_id"], local_dir=str(target))
                if "adapter_file" in entry:
                    src = hf_hub_download(repo_id=entry["adapter_repo"], repo_type="dataset", filename=entry["adapter_file"])
                    (target / "instruction_emb.pt").write_bytes(Path(src).read_bytes())
    elif args.what == "helpers":
        for model in ("mistral_7b", "llama2_7b"):
            print(f"[fetch] {models[model]['helper']['hf_id']} -> HF cache")
            snapshot_download(repo_id=models[model]["helper"]["hf_id"])
    else:
        for entry in (models["llama31_8b"]["checkpoints"]["perinucleus_basecontrol"], models["llama31_8b"]["helper"]):
            target = root / entry["path"]
            print(f"[fetch] {entry['download']['repo']}@{entry['download']['revision']} -> {target}")
            snapshot_download(repo_id=entry["download"]["repo"], revision=entry["download"]["revision"],
                              local_dir=str(target), allow_patterns=["*.json", "*.safetensors", "original/tokenizer.model"])


if __name__ == "__main__":
    main()
