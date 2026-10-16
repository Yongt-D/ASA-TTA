"""AP of oriented boxes at IoU 0.50:0.05:0.95 (101-point interpolation, class mean).

Predictions are matched to ground truth per class by polygon IoU, greedily in order of
decreasing confidence, each ground-truth box at most once; at most ``max_dets`` predictions
per image. Only images with a label file are evaluated.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .obb import fast_iou, read_obb_file

THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]


def ap_from_pr(tp, fp, n_gt):
    if n_gt <= 0:
        return None
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / max(n_gt, 1)
    precision = tp / np.maximum(tp + fp, 1e-12)
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        vals = precision[recall >= thr]
        ap += (vals.max() if vals.size else 0.0) / 101.0
    return float(ap)


def _load(label_dir, pred_dir, num_classes, max_dets):
    gt_count = [0] * num_classes
    gt_by_img_cls = {}
    pred_by_cls = [[] for _ in range(num_classes)]
    for label_path in sorted(Path(label_dir).glob("*.txt")):
        image_id = label_path.stem
        by_cls = defaultdict(list)
        for gt in read_obb_file(label_path, is_pred=False):
            cls = int(gt["cls"])
            if 0 <= cls < num_classes:
                gt_count[cls] += 1
                by_cls[cls].append(gt)
        gt_by_img_cls[image_id] = dict(by_cls)
        preds = sorted(read_obb_file(Path(pred_dir) / label_path.name, is_pred=True),
                       key=lambda p: p["conf"], reverse=True)[:max_dets]
        for pred in preds:
            cls = int(pred["cls"])
            if 0 <= cls < num_classes:
                pred_by_cls[cls].append((image_id, float(pred["conf"]), pred))
    return gt_count, gt_by_img_cls, pred_by_cls


def _one(args):
    cls, thr, gt_n, gt_by_img_cls, preds = args
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    used = defaultdict(set)
    tp, fp = [], []
    for image_id, _, pred in preds:
        gts = gt_by_img_cls.get(image_id, {}).get(cls, [])
        best_iou, best_idx = 0.0, -1
        for idx, gt in enumerate(gts):
            if idx in used[image_id]:
                continue
            iou = fast_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        hit = best_iou >= thr and best_idx >= 0
        if hit:
            used[image_id].add(best_idx)
        tp.append(1.0 if hit else 0.0)
        fp.append(0.0 if hit else 1.0)
    return cls, thr, ap_from_pr(np.asarray(tp), np.asarray(fp), gt_n)


def evaluate(label_dir, pred_dir, num_classes, max_dets=300, workers=8):
    """{'mAP50', 'AP75', 'mAP50_95', 'mAP_by_iou'} as fractions in [0, 1]."""
    gt_count, gt_by_img_cls, pred_by_cls = _load(label_dir, pred_dir, num_classes, max_dets)
    jobs = [(cls, thr, gt_count[cls], gt_by_img_cls, pred_by_cls[cls])
            for cls in range(num_classes) if gt_count[cls] > 0 for thr in THRESHOLDS]
    if workers <= 1:
        rows = [_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_one, jobs))
    by_thr = {f"{thr:.2f}": {} for thr in THRESHOLDS}
    for cls, thr, ap in rows:
        by_thr[f"{thr:.2f}"][str(cls)] = float(ap)
    m = {k: float(np.mean(list(v.values()))) if v else 0.0 for k, v in by_thr.items()}
    return {"mAP50": m["0.50"], "AP75": m["0.75"], "mAP50_95": float(np.mean(list(m.values()))), "mAP_by_iou": m}


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="AP50, AP75 and AP50:95 (percent) of one prediction directory")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--max-dets", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    r = evaluate(a.labels, a.preds, a.num_classes, a.max_dets, a.workers)
    print(json.dumps({k: round(100 * r[k], 2) for k in ("mAP50", "AP75", "mAP50_95")}))


if __name__ == "__main__":
    main()
