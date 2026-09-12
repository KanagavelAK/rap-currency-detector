"""Fine-tune RT-DETR and record everything needed to reproduce the run.

Usage:
  # 1-epoch smoke test on 10% of the data (catches path/label bugs in minutes)
  python scripts/train.py --data datasets/currency/data_public.yaml --name smoke --smoke

  # Run A: public data only
  python scripts/train.py --data datasets/currency/data_public.yaml --name run_a

  # Run B: public + your own training sessions
  python scripts/train.py --data datasets/currency/data_plus_own.yaml --name run_b
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu for CPU")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="1 epoch on 10%% of the training data")
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import RTDETR

    cfg = yaml.safe_load(args.config.read_text())
    model_name = cfg.pop("model")
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.smoke:
        cfg.update(epochs=1, fraction=0.1, patience=0, plots=False)
    if args.device is not None:
        cfg["device"] = args.device

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"Training {model_name} on {gpu} with config: {cfg}")

    model = RTDETR(model_name)
    start = time.time()
    model.train(data=str(args.data), project=args.project, name=args.name, **cfg)
    minutes = (time.time() - start) / 60

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    copied = None
    if best.exists():
        copied = weights_dir / f"{args.name}_best.pt"
        shutil.copy2(best, copied)

    info = {
        "run_name": args.name,
        "data_yaml": str(args.data),
        "base_model": model_name,
        "train_args": cfg,
        "wall_clock_minutes": round(minutes, 1),
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "save_dir": str(save_dir),
        "best_weights": str(copied) if copied else None,
    }
    (save_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    if not best.exists():
        sys.exit("WARNING: best.pt not found - check the training log above.")


if __name__ == "__main__":
    main()
