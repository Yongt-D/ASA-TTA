"""Export the predictions of an MMRotate detector (Oriented R-CNN) under TTA views.

The MMRotate counterpart of export_views_yolo.py, writing the same files:

    python tools/export_views_mmrotate.py --config configs/mmrotate/oriented_rcnn_r50_dota.py \
        --checkpoint work_dirs/.../epoch_12.pth --source data/dota_yolo_obb_1024/images/val \
        --classes data/mmrotate/dota/classes.json --out-root views/dota_orcnn/val/base \
        --views r0 r90 r180 r270 h0 h90 h180 h270
    ... --out-root views/dota_orcnn/val/0.83 --img-scale 832 --views h0

Class indices follow ``classes.json`` written by yolo_obb_to_dota_annfiles.py, matched by name.
The model's score threshold is lowered to the export floor 0.001 and at most 1000 boxes are
kept. ``--img-scale`` sets the square test Resize, for the multi-scale views. Requires
MMRotate 1.x (dev-1.x) with mmengine and mmdet.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.obb import IMAGE_SUFFIXES  # noqa: E402
from asa.views import BACK_MAPPING, inverse_view_points, parse_view, view_image  # noqa: E402


def write_predictions(path, corners, labels, scores, k, flip, width, height, label_map):
    path.parent.mkdir(parents=True, exist_ok=True)
    if corners is None or len(corners) == 0:
        path.write_text("")
        return 0
    polygons = inverse_view_points(corners.reshape(-1, 4, 2), k, flip, width, height)
    polygons[..., 0] = np.clip(polygons[..., 0] / width, 0.0, 1.0)
    polygons[..., 1] = np.clip(polygons[..., 1] / height, 0.0, 1.0)
    lines = [f"{label_map[int(c)]} " + " ".join(f"{v:.8g}" for v in p.reshape(-1)) + f" {float(s):.8g}"
             for c, p, s in zip(labels, polygons, scores)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--classes", required=True, help="classes.json from yolo_obb_to_dota_annfiles.py")
    ap.add_argument("--views", nargs="+", default=["r0", "r90", "r180", "r270"])
    ap.add_argument("--img-scale", type=int, default=None, help="square test Resize, e.g. 832")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only-labelled", metavar="LABEL_DIR",
                    help="export only images with a label file here (only those are evaluated)")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    init_default_scope("mmrotate")
    from mmdet.apis import inference_detector, init_detector
    from mmrotate.structures.bbox import rbox2qbox

    cfg = Config.fromfile(a.config)
    if not cfg.get("test_dataloader"):
        # the training configs switch the test loop off; init_detector still builds a dataset
        # from test_dataloader.dataset, so reuse the train dataset with the test pipeline
        ds = cfg.train_dataloader.dataset.copy()
        ds.update(pipeline=cfg.test_pipeline, test_mode=True)
        cfg.test_dataloader = dict(dataset=ds)
    if a.img_scale is not None:
        for step in cfg.test_dataloader.dataset.pipeline:
            if step["type"] in ("mmdet.Resize", "Resize"):
                step["scale"] = (a.img_scale, a.img_scale)
    model = init_detector(cfg, a.checkpoint, device=a.device)
    test_cfg = model.test_cfg["rcnn"] if "rcnn" in model.test_cfg else model.test_cfg
    test_cfg["score_thr"] = a.conf
    test_cfg["max_per_img"] = a.max_det

    ours = {int(k): v for k, v in json.loads(Path(a.classes).read_text()).items()}
    by_token = {v: k for k, v in ours.items()}
    names = list(model.dataset_meta["classes"])
    if any(n not in by_token for n in names) or len(names) != len(ours):
        raise SystemExit(f"model classes {names} do not match {a.classes}")
    label_map = {i: by_token[n] for i, n in enumerate(names)}

    views = [(v, *parse_view(v)) for v in a.views]
    out_root = Path(a.out_root)
    dirs = [out_root / v for v, _, _ in views]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(a.source).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if a.only_labelled:
        labelled = {q.stem for q in Path(a.only_labelled).glob("*.txt")}
        paths = [p for p in paths if p.stem in labelled]
    if a.skip_existing:
        paths = [p for p in paths if not all((d / f"{p.stem}.txt").exists() for d in dirs)]
    totals = [0] * len(views)
    for n, path in enumerate(paths, start=1):
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"failed to read {path}")
        height, width = original.shape[:2]
        for i, (name, k, flip) in enumerate(views):
            inst = inference_detector(model, np.ascontiguousarray(view_image(original, k, flip))).pred_instances
            boxes = inst.bboxes
            corners = rbox2qbox(boxes.tensor if hasattr(boxes, "tensor") else boxes).detach().cpu().numpy()
            totals[i] += write_predictions(dirs[i] / f"{path.stem}.txt", corners, inst.labels.detach().cpu().numpy(),
                                           inst.scores.detach().cpu().numpy(), k, flip, width, height, label_map)
        if n % 100 == 0 or n == len(paths):
            print(f"exported {n}/{len(paths)} images, boxes per view {totals}", flush=True)

    import mmrotate
    (out_root / "export.json").write_text(json.dumps({
        "back_mapping": BACK_MAPPING, "framework": f"mmrotate {mmrotate.__version__}", "config": str(a.config),
        "checkpoint": str(a.checkpoint), "source": str(a.source), "img_scale": a.img_scale, "conf": a.conf,
        "max_det": a.max_det, "views": a.views, "totals": dict(zip(a.views, totals)),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))


if __name__ == "__main__":
    main()
