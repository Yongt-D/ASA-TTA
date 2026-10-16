"""GPU latency of each TTA family relative to one r0 pass, YOLO11-OBB, batch size 1 (Table 10).

    python tools/benchmark_views.py --model yolo11s-obb.pt --source data/dota_yolo_obb_1024/images/val \
        --imgsz 1536 --n 100 --repeats 3 --out results/latency/views_yolo11s_1536.json

Images are decoded into memory first, so disk I/O is excluded. For every image and family the
clock covers building each view, the detector pass at the view's input size (Ultralytics
predict: pre-processing, inference, NMS; confidence 0.001, at most 1000 boxes, as in the
exports) and mapping the polygons back; rotated WBF (0.55, max score, 300 boxes) is timed
separately, since it is a pure-Python implementation and says more about the code than about
the family. Per repeat the per-image times are reduced to a median, and the report gives the
median over repeats. Families are interleaved per image in a rotating order, so that slow
drifts of the GPU spread over all of them.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.fusion import classwise_wbf  # noqa: E402
from asa.obb import IMAGE_SUFFIXES, poly_bounds  # noqa: E402
from asa.views import FAMILIES, inverse_view_points, parse_view, view_image  # noqa: E402

FAMS = {"r0": ["r0"]} | FAMILIES


def view_size(base, scale):
    """Input size of a scaled pass, as exported (1536 -> 1280 / 1024, 1024 -> 832 / 672)."""
    if scale == 1.0:
        return base
    table = {1536: {0.83: 1280, 0.67: 1024}, 1024: {0.83: 832, 0.67: 672}}
    return table.get(base, {}).get(scale, int(round(base * scale / 32)) * 32)


def run_family(model, image, views, base, wbf, device):
    import torch
    sync = torch.cuda.synchronize if device != "cpu" else (lambda: None)
    h, w = image.shape[:2]
    sync()
    t0 = time.perf_counter()
    preds = []
    for name in views:
        k, flip = parse_view(name)
        scale = float(name.partition("@")[2] or 1.0)
        r = model.predict(np.ascontiguousarray(view_image(image, k, flip)), imgsz=view_size(base, scale),
                          conf=0.001, iou=0.7, max_det=1000, device=device, verbose=False)[0]
        if r.obb is None or len(r.obb) == 0:
            continue
        poly = inverse_view_points(r.obb.xyxyxyxy.cpu().numpy(), k, flip, w, h)
        poly[..., 0] /= w
        poly[..., 1] /= h
        preds.append((poly, r.obb.cls.cpu().numpy().astype(int), r.obb.conf.cpu().numpy()))
    sync()
    t1 = time.perf_counter()
    if wbf and len(views) > 1:
        items = []
        for poly, cls, conf in preds:
            for pl, c, s in zip(poly, cls, conf):
                pts = [tuple(p) for p in pl]
                items.append({"cls": int(c), "poly": pts, "bounds": poly_bounds(pts), "conf": float(s)})
        classwise_wbf(items, 0.55, "max")[:300]
    return t1 - t0, time.perf_counter() - t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--imgsz", type=int, required=True, help="base input size")
    ap.add_argument("--n", type=int, default=100, help="images, evenly spaced over the sorted set")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--families", nargs="+", default=list(FAMS), choices=list(FAMS))
    ap.add_argument("--no-wbf", action="store_true", help="time the views only")
    ap.add_argument("--device", default="0", help="CUDA device index, or cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO

    paths = sorted(p for p in Path(a.source).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    images = [cv2.imread(str(p)) for p in paths[::max(1, len(paths) // a.n)][:a.n]]
    model = YOLO(a.model)
    sizes = {view_size(a.imgsz, float(v.partition("@")[2] or 1.0)) for f in a.families for v in FAMS[f]}
    for img in images[:a.warmup]:  # cuDNN autotuning and lazy initialisation, at every input size
        for s in sizes:
            model.predict(img, imgsz=s, conf=0.001, iou=0.7, max_det=1000, device=a.device, verbose=False)

    fams = a.families
    per = {f: {"views": [], "fusion": []} for f in fams}
    for rep in range(a.repeats):
        times = {f: ([], []) for f in fams}
        for i, img in enumerate(images):
            for f in fams[i % len(fams):] + fams[:i % len(fams)]:
                tv, tf = run_family(model, img, FAMS[f], a.imgsz, not a.no_wbf, a.device)
                times[f][0].append(tv)
                times[f][1].append(tf)
        for f in fams:
            per[f]["views"].append(statistics.median(times[f][0]))
            per[f]["fusion"].append(statistics.median(times[f][1]))
        print(f"repeat {rep + 1}/{a.repeats}", flush=True)

    med = {f: {k: statistics.median(v) for k, v in per[f].items()} for f in fams}
    base = med["r0"]["views"] if "r0" in med else None
    report = {
        "model": a.model, "source": a.source, "imgsz": a.imgsz, "images": len(images), "repeats": a.repeats,
        "batch": 1, "gpu": torch.cuda.get_device_name(int(a.device)) if a.device != "cpu" else "cpu",
        "torch": torch.__version__,
        "note": "median over images per repeat, then median over repeats; seconds",
        "families": {f: {"passes": len(FAMS[f]), "views_s": med[f]["views"], "fusion_s": med[f]["fusion"],
                         "views_rel": med[f]["views"] / base if base else None,
                         "total_rel": (med[f]["views"] + med[f]["fusion"]) / base if base else None,
                         "per_repeat_views_s": per[f]["views"]} for f in fams},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    for f in fams:
        r = report["families"][f]
        rel = f" ({r['views_rel']:.2f}x)" if base else ""
        print(f"{f:5s} passes {r['passes']}  views {1000 * r['views_s']:7.1f} ms{rel}  fusion {1000 * r['fusion_s']:7.1f} ms")


if __name__ == "__main__":
    main()
