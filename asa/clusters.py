"""Rotated-WBF clusters of the pooled views, their descriptors, and writing rescored clusters.

``build_members`` turns the view predictions of one split into the arrays that every
aggregation rule reads: per cluster the cluster descriptor ``X`` (``cluster_features``), the
WBF box, the class, the image and its aspect ratio, and (for training) the best IoU with a
same-class ground-truth box and that box; per member (the ``M_MAX`` most confident) its view,
confidence, polygon and IoU with the WBF box.
"""
import math
from pathlib import Path

import numpy as np

from .fusion import weighted_box
from .obb import fast_iou, image_aspects, poly_bounds, read_obb_file, write_obb_file

M_MAX = 8        # members kept per cluster for the set encoder
CLASS_COL = 14   # column of the class index in the cluster descriptor


def _side_lengths(poly):
    return [math.dist(poly[i], poly[(i + 1) % 4]) for i in range(4)]


def clusters_for_image(name, view_dirs, views, iou_thr=0.55):
    """Rotated-WBF clusters of one image: list of (WBF box with max score, members)."""
    by_cls = {}
    for d, v in zip(view_dirs, views):
        for p in read_obb_file(Path(d) / name, is_pred=True):
            p = dict(p)
            p["view"] = v
            p["conf"] = max(float(p["conf"]), 1e-9)
            by_cls.setdefault(int(p["cls"]), []).append(p)
    out = []
    for preds in by_cls.values():
        remaining = sorted(preds, key=lambda p: p["conf"], reverse=True)
        while remaining:
            seed = remaining.pop(0)
            cluster, keep = [seed], []
            for p in remaining:
                (cluster if fast_iou(seed, p) >= iou_thr else keep).append(p)
            remaining = keep
            out.append((weighted_box(cluster, "max"), cluster))
    return out


def cluster_features(box, members, views):
    """The cluster descriptor: 15 base features (column 14 = class) and one presence flag per view.

    0 distinct views, 1 their fraction, 2 members, 3-6 max/mean/min/std of the confidences,
    7 highest confidence of an r0 member (0 if none), 8 r0 present, 9 WBF max score (equal to
    column 3), 10-11 mean/min IoU of the members with the WBF box, 12 log area and 13 aspect
    ratio of the WBF box in normalised image coordinates (the network takes the log of 13),
    14 class.
    """
    conf = np.array([m["conf"] for m in members])
    present = {m["view"] for m in members}
    r0 = [m["conf"] for m in members if m["view"] == "r0"]
    ious = [fast_iou(box, m) for m in members]
    s = _side_lengths(box["poly"])
    w, h = max(s[0], s[2], 1e-9), max(s[1], s[3], 1e-9)
    base = [len(present), len(present) / len(views), len(members), conf.max(), conf.mean(),
            conf.min(), conf.std(), max(r0) if r0 else 0.0, 1.0 if r0 else 0.0, box["conf"],
            float(np.mean(ious)), float(np.min(ious)), math.log(w * h), max(w, h) / min(w, h),
            float(box["cls"])]
    return base + [1.0 if v in present else 0.0 for v in views]


def build_members(label_dir, img_dir, view_dirs, views, iou_thr=0.55, aspects=None, with_labels=True,
                  names=None):
    """Cluster and member arrays for every image of a split (one image per label file).

    ``aspects`` maps image stems to height / width (read from ``img_dir`` when not given);
    ``with_labels=False`` skips the ground truth, i.e. computes only what inference needs;
    ``names`` (prediction file names, e.g. ``P0003.txt``) replaces the label files as the
    list of images, for images without labels.
    """
    files = list(names) if names is not None else [p.name for p in sorted(Path(label_dir).glob("*.txt"))]
    if aspects is None:
        aspects = image_aspects(img_dir, [Path(f).stem for f in files])
    C = {k: [] for k in ("X", "y", "fpoly", "gpoly", "cls", "name", "aspect")}
    Mv, Mc, Mp, Mi, Mn = [], [], [], [], []
    seen = []
    vidx = {v: i for i, v in enumerate(views)}
    for name in files:
        gts = read_obb_file(Path(label_dir) / name, is_pred=False) if with_labels else []
        ni = len(seen)
        seen.append(name)
        asp = aspects.get(Path(name).stem, 1.0)
        for box, members in clusters_for_image(name, view_dirs, views, iou_thr):
            C["X"].append(cluster_features(box, members, views))
            best, gpoly = 0.0, [0.0] * 8
            for g in gts:
                if int(g["cls"]) == int(box["cls"]):
                    iou = fast_iou(box, g)
                    if iou > best:
                        best, gpoly = iou, [c for pt in g["poly"] for c in pt]
            C["y"].append(best)
            C["gpoly"].append(gpoly)
            C["fpoly"].append([c for pt in box["poly"] for c in pt])
            C["cls"].append(int(box["cls"]))
            C["name"].append(ni)
            C["aspect"].append(asp)
            ms = sorted(members, key=lambda m: m["conf"], reverse=True)[:M_MAX]
            Mn.append(len(ms))
            for m in ms:
                Mv.append(vidx[m["view"]])
                Mc.append(m["conf"])
                Mp.append([c for pt in m["poly"] for c in pt])
                Mi.append(fast_iou(box, m))
    out = {k: np.asarray(v, dtype=np.float32 if k not in ("cls", "name") else np.int64) for k, v in C.items()}
    out["X"] = np.asarray(C["X"], dtype=np.float64).reshape(len(C["y"]), -1)
    out.update(mview=np.asarray(Mv, np.int64), mconf=np.asarray(Mc, np.float32),
               mpoly=np.asarray(Mp, np.float32).reshape(-1, 8), miou=np.asarray(Mi, np.float32),
               mcount=np.asarray(Mn, np.int64), names=np.asarray(seen))
    return out


def boxes_from(z, polys):
    """(image file, box dict) per cluster, in cluster order."""
    out = []
    for n, p, c in zip(z["name"], polys, z["cls"]):
        poly = [(float(p[i]), float(p[i + 1])) for i in range(0, 8, 2)]
        out.append((str(z["names"][n]), {"cls": int(c), "poly": poly, "bounds": poly_bounds(poly), "conf": 0.0}))
    return out


def write_rescored(out_dir, boxes, scores, label_dir, max_dets=1000, names=None):
    """One prediction file per label file (or per entry of ``names``), boxes sorted by the new
    scores, at most ``max_dets``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per = {}
    for (name, box), s in zip(boxes, scores):
        b = dict(box)
        b["conf"] = float(s)
        per.setdefault(name, []).append(b)
    files = list(names) if names is not None else [p.name for p in sorted(Path(label_dir).glob("*.txt"))]
    for name in files:
        items = sorted(per.get(name, []), key=lambda b: b["conf"], reverse=True)[:max_dets]
        write_obb_file(out_dir / name, items)
