"""Paired bootstrap of AP50 differences with resamples shared across the cells of a dataset.

Each resample draws images (or, where crops share a source scene, scenes with all their crops)
with replacement; the same resamples are applied to every prediction set of the dataset, so
dataset-level statistics such as the mean over cells of AP50(A) - AP50(B) are resampled
jointly. Matching is the evaluator's (``asa.evaluate``) and is computed once per image: a
resample only repeats images.
"""
import re
from pathlib import Path

import numpy as np

from .obb import fast_iou, read_obb_file

THR = np.linspace(0, 1, 101)


def match_source(label_dir, pred_dir, num_classes, max_dets=300):
    """Flat arrays over all predictions: image index, class, TP flag, plus GT counts per image."""
    img, cls, conf, tp = [], [], [], []
    labels = sorted(Path(label_dir).glob("*.txt"))
    gt_count = np.zeros((len(labels), num_classes), np.int32)
    for i, lab in enumerate(labels):
        gts = read_obb_file(lab, is_pred=False)
        preds = sorted(read_obb_file(Path(pred_dir) / lab.name, is_pred=True),
                       key=lambda p: p["conf"], reverse=True)[:max_dets]
        for k in range(num_classes):
            g = [x for x in gts if int(x["cls"]) == k]
            gt_count[i, k] = len(g)
            used = set()
            for p in (x for x in preds if int(x["cls"]) == k):
                best, bi = 0.0, -1
                for j, x in enumerate(g):
                    if j in used:
                        continue
                    v = fast_iou(p, x)
                    if v > best:
                        best, bi = v, j
                hit = best >= 0.5 and bi >= 0
                if hit:
                    used.add(bi)
                img.append(i), cls.append(k), conf.append(float(p["conf"])), tp.append(hit)
    img, cls, conf, tp = (np.asarray(img, np.int32), np.asarray(cls, np.int16),
                          np.asarray(conf, np.float64), np.asarray(tp, bool))
    order = np.lexsort((-conf, cls))  # by class, then confidence descending (stable)
    return {"img": img[order], "cls": cls[order], "tp": tp[order], "gt": gt_count}


def ap50(src, w, num_classes):
    """Class-mean AP50 (101-point) when image i is drawn w[i] times."""
    rep = w[src["img"]]
    tp = np.repeat(src["tp"], rep)
    cls = np.repeat(src["cls"], rep)
    ngt = (src["gt"] * w[:, None]).sum(0)
    bounds = np.searchsorted(cls, np.arange(num_classes + 1))
    aps = []
    for k in range(num_classes):
        if ngt[k] <= 0:
            continue
        t = tp[bounds[k]:bounds[k + 1]]
        if t.size == 0:
            aps.append(0.0)
            continue
        ctp = np.cumsum(t)
        prec = ctp / np.arange(1, t.size + 1)
        rec = ctp / ngt[k]
        env = np.maximum.accumulate(prec[::-1])[::-1]
        idx = np.searchsorted(rec, THR, side="left")
        aps.append(float(np.where(idx < t.size, env[np.minimum(idx, t.size - 1)], 0.0).sum() / 101))
    return float(np.mean(aps)) if aps else np.nan


def resample_weights(label_dir, repeats=1000, seed=3086, scene_pattern=None, per_repeat_seed=False):
    """[repeats, n_images] multiplicities; with ``scene_pattern`` the unit is the scene
    (first group of the regex on the label file name), otherwise the image.

    Resample b draws from one random sequence started at ``seed``, or, with
    ``per_repeat_seed``, from its own generator seeded ``seed + b`` (the scheme of the
    family-versus-single-view contrasts of Table 1)."""
    names = [p.name for p in sorted(Path(label_dir).glob("*.txt"))]
    if scene_pattern:
        rx = re.compile(scene_pattern)
        units = [rx.match(n).group(1) if rx.match(n) else n for n in names]
    else:
        units = names
    keys = sorted(set(units))
    index = {k: i for i, k in enumerate(keys)}
    kid = np.array([index[u] for u in units])
    rng = np.random.default_rng(seed)
    W = np.zeros((repeats, len(names)), np.int64)
    for b in range(repeats):
        g = np.random.default_rng(seed + b) if per_repeat_seed else rng
        W[b] = np.bincount(g.integers(0, len(keys), len(keys)), minlength=len(keys))[kid]
    return W, len(keys)


def percentile_ci(x, level=0.95):
    a = (1 - level) / 2
    return [float(np.nanquantile(x, a)), float(np.nanquantile(x, 1 - a))]
