"""Export the predictions of a YOLO11-OBB checkpoint under TTA views, mapped back to the image.

    python tools/export_views_yolo.py --model best.pt --source data/.../images/val \
        --out-root views/hrsc_yolo11s/val/base --imgsz 1024 --views r0 r90 r180 r270 h0 h90 h180 h270
    python tools/export_views_yolo.py ... --out-root views/hrsc_yolo11s/val/0.83 --imgsz 832 --views h0
    python tools/export_views_yolo.py ... --out-root views/hrsc_yolo11s/val/0.67 --imgsz 672 --views r0

One file per image and view, ``cls x1 y1 ... x4 y4 conf`` with vertices normalised by the
original image size and clipped to [0, 1]; confidence floor 0.001 and at most 1000 boxes, so
that ranking is not cut before fusion. ``export.json`` records the back-mapping. Export the
training images in the same way: they are the training data of the aggregator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.obb import IMAGE_SUFFIXES  # noqa: E402
from asa.views import BACK_MAPPING, inverse_view_points, parse_view, view_image  # noqa: E402


def write_predictions(path: Path, result, k: int, flip: bool, width: int, height: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if result.obb is None or len(result.obb) == 0:
        path.write_text("")
        return 0
    polygons = inverse_view_points(result.obb.xyxyxyxy.detach().cpu().numpy(), k, flip, width, height)
    polygons[..., 0] = np.clip(polygons[..., 0] / width, 0.0, 1.0)
    polygons[..., 1] = np.clip(polygons[..., 1] / height, 0.0, 1.0)
    classes = result.obb.cls.detach().cpu().numpy().astype(int)
    confidences = result.obb.conf.detach().cpu().numpy()
    lines = [f"{c} " + " ".join(f"{v:.8g}" for v in p.reshape(-1)) + f" {float(s):.8g}"
             for c, p, s in zip(classes, polygons, confidences)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main() -> None:
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, help="image directory")
    ap.add_argument("--out-root", required=True, help="one sub-directory per view is created here")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--views", nargs="+", default=["r0", "r90", "r180", "r270"],
                    help="r{0,90,180,270} rotate; h{...} rotate, then flip left-right")
    ap.add_argument("--only-labelled", metavar="LABEL_DIR",
                    help="export only images with a label file here (only those are evaluated)")
    ap.add_argument("--skip-existing", action="store_true", help="resume: skip images done in every view")
    a = ap.parse_args()

    paths = sorted(p for p in Path(a.source).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"no images under {a.source}")
    out_root = Path(a.out_root)
    views = [(name, *parse_view(name)) for name in a.views]
    dirs = [out_root / name for name, _, _ in views]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    if a.only_labelled:
        labelled = {q.stem for q in Path(a.only_labelled).glob("*.txt")}
        paths = [p for p in paths if p.stem in labelled]
    if a.skip_existing:
        paths = [p for p in paths if not all((d / f"{p.stem}.txt").exists() for d in dirs)]

    model = YOLO(a.model)
    totals = [0] * len(views)
    for start in range(0, len(paths), a.batch):
        chunk = paths[start:start + a.batch]
        originals = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in chunk]
        if any(im is None for im in originals):
            raise RuntimeError(f"failed to read {[str(p) for p, im in zip(chunk, originals) if im is None]}")
        for v, (name, k, flip) in enumerate(views):
            views_in = [np.ascontiguousarray(view_image(im, k, flip)) for im in originals]
            results = model.predict(source=views_in, imgsz=a.imgsz, batch=len(views_in), conf=a.conf,
                                    iou=a.iou, max_det=a.max_det, device=a.device, verbose=False)
            for p, im, r in zip(chunk, originals, results):
                h, w = im.shape[:2]
                totals[v] += write_predictions(dirs[v] / f"{p.stem}.txt", r, k, flip, w, h)
        done = min(start + len(chunk), len(paths))
        if done % 100 == 0 or done == len(paths):
            print(f"exported {done}/{len(paths)} images, boxes per view {totals}", flush=True)
    (out_root / "export.json").write_text(json.dumps({
        "back_mapping": BACK_MAPPING, "model": a.model, "source": str(a.source), "imgsz": a.imgsz,
        "conf": a.conf, "iou": a.iou, "max_det": a.max_det, "images": len(paths), "views": a.views,
        "totals": dict(zip(a.views, totals))}, indent=2))


if __name__ == "__main__":
    main()
