"""Train a YOLO11-OBB detector by replaying its recorded training arguments.

    python tools/train_yolo11.py configs/yolo11/hrsc_yolo11s.yaml --project work_dirs/yolo11
    python tools/train_yolo11.py configs/yolo11/hrsc_yolo11s.yaml --resume

The yaml holds everything Ultralytics recorded for the run (recipe, augmentation, schedule,
seed) except the machine-specific entries, which come from the command line. ``model`` names
the released Ultralytics OBB checkpoint the run was fine-tuned from (trained on DOTA-v1.0,
downloaded by Ultralytics) and ``data`` the dataset yaml written by the converters. Training is not bit-reproducible across
GPUs (``deterministic: false``), so a retrained detector differs slightly from ours.
"""
import argparse
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--project", default="work_dirs/yolo11")
    ap.add_argument("--name", help="run name (default: the config's stem)")
    ap.add_argument("--data", help="override the dataset yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="resume from <project>/<name>/weights/last.pt")
    a = ap.parse_args()

    from ultralytics import YOLO
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    name = a.name or Path(a.config).stem
    last = Path(a.project) / name / "weights" / "last.pt"
    if a.resume:
        if not last.exists():
            raise SystemExit(f"nothing to resume: {last} does not exist")
        YOLO(str(last)).train(resume=True)
        return
    model = cfg.pop("model")
    data = a.data or cfg.pop("data")
    cfg.pop("data", None)
    YOLO(model).train(**cfg, data=data, project=str(Path(a.project).resolve()), name=name,
                      device=a.device, workers=a.workers, exist_ok=True)


if __name__ == "__main__":
    main()
