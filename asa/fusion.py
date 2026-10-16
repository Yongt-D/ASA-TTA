"""Fixed fusion rules for the pooled predictions of the views: rotated WBF, NMS and Soft-NMS.

All rules work per image and per class with polygon IoU. Rotated WBF seeds a cluster with the
most confident remaining box, adds every remaining box whose IoU with the seed is at least
``iou_thr``, and replaces the cluster by the confidence-weighted mean of its members' vertices
after aligning each member's vertex order to the seed.
"""
import math

from .obb import fast_iou, poly_bounds, read_obb_file


def align_poly(poly, ref):
    """The cyclic shift or reversal of ``poly``'s vertices closest to ``ref`` in squared distance."""
    candidates = []
    for pts in (poly, list(reversed(poly))):
        for shift in range(4):
            cand = pts[shift:] + pts[:shift]
            cost = sum((cand[i][0] - ref[i][0]) ** 2 + (cand[i][1] - ref[i][1]) ** 2 for i in range(4))
            candidates.append((cost, cand))
    return min(candidates, key=lambda x: x[0])[1]


def weighted_box(cluster, conf_type="max"):
    ref = max(cluster, key=lambda p: p["conf"])
    weights = [max(float(p["conf"]), 1e-9) for p in cluster]
    total_w = sum(weights)
    aligned = [align_poly(p["poly"], ref["poly"]) for p in cluster]
    poly = []
    for i in range(4):
        x = sum(points[i][0] * w for points, w in zip(aligned, weights)) / total_w
        y = sum(points[i][1] * w for points, w in zip(aligned, weights)) / total_w
        poly.append((x, y))
    if conf_type == "max":
        conf = max(float(p["conf"]) for p in cluster)
    elif conf_type == "mean":
        conf = sum(float(p["conf"]) for p in cluster) / len(cluster)
    else:
        raise ValueError(f"unknown conf_type {conf_type}")
    return {"cls": int(ref["cls"]), "poly": poly, "bounds": poly_bounds(poly), "conf": conf}


def wbf_class(preds, iou_thr, conf_type="max"):
    remaining = sorted([dict(p) for p in preds], key=lambda p: p["conf"], reverse=True)
    fused = []
    while remaining:
        seed = remaining.pop(0)
        cluster, keep = [seed], []
        for pred in remaining:
            (cluster if fast_iou(seed, pred) >= iou_thr else keep).append(pred)
        fused.append(weighted_box(cluster, conf_type))
        remaining = keep
    return fused


def classwise_wbf(preds, iou_thr=0.55, conf_type="max"):
    by_cls = {}
    for pred in preds:
        by_cls.setdefault(int(pred["cls"]), []).append(pred)
    fused = []
    for cls_preds in by_cls.values():
        fused.extend(wbf_class(cls_preds, iou_thr, conf_type))
    return sorted(fused, key=lambda p: p["conf"], reverse=True)


def classwise_rotated_nms(preds, iou_thr=0.55):
    kept, by_cls = [], {}
    for pred in preds:
        by_cls.setdefault(int(pred["cls"]), []).append(pred)
    for cls_preds in by_cls.values():
        cls_preds = sorted(cls_preds, key=lambda p: p["conf"], reverse=True)
        while cls_preds:
            best = cls_preds.pop(0)
            kept.append(best)
            cls_preds = [p for p in cls_preds if fast_iou(best, p) < iou_thr]
    return sorted(kept, key=lambda p: p["conf"], reverse=True)


def _soft_nms_class(preds, iou_thr, sigma, score_thresh):
    preds = [dict(p) for p in preds]
    kept = []
    while preds:
        best_idx = max(range(len(preds)), key=lambda i: preds[i]["conf"])
        best = preds.pop(best_idx)
        kept.append(best)
        updated = []
        for pred in preds:
            iou = fast_iou(best, pred)
            if iou >= iou_thr:
                pred["conf"] = pred["conf"] * math.exp(-(iou * iou) / sigma) if sigma > 0 else 0.0
            if pred["conf"] >= score_thresh:
                updated.append(pred)
        preds = updated
    return kept


def classwise_soft_nms(preds, iou_thr=0.55, sigma=0.5, score_thresh=1e-6):
    """Gaussian Soft-NMS; only boxes overlapping the kept box by at least ``iou_thr`` are decayed."""
    kept, by_cls = [], {}
    for pred in preds:
        by_cls.setdefault(int(pred["cls"]), []).append(pred)
    for cls_preds in by_cls.values():
        kept.extend(_soft_nms_class(cls_preds, iou_thr, sigma, score_thresh))
    return sorted(kept, key=lambda p: p["conf"], reverse=True)


def pooled_predictions(name, view_dirs):
    """All boxes of one image over the given view directories."""
    preds = []
    for d in view_dirs:
        for pred in read_obb_file(d / name, is_pred=True):
            pred = dict(pred)
            pred["conf"] = max(float(pred["conf"]), 1e-9)
            preds.append(pred)
    return preds


RULES = {
    "wbf_max": lambda p, t: classwise_wbf(p, t, "max"),
    "wbf_mean": lambda p, t: classwise_wbf(p, t, "mean"),
    "nms": classwise_rotated_nms,
    "softnms": classwise_soft_nms,
}
