"""Qualitative analysis: WBF with the original score against ASA (Section 6.6 and Figure 3).

    python tools/qualitative.py stats
    python tools/qualitative.py figure --out figure_qualitative.pdf

Reads the rescored predictions of tools/run_cell.py ({preds}/<detector>/<family>/{wbf_orig,asa}).
Both rules keep, over the evaluation split of a cell, as many of their highest-scoring boxes
(top 300 per image, as evaluated) as there are ground-truth objects. A kept box is a true
positive if it matches a ground-truth box of its class at IoU >= 0.5, greedily by score.
False positives are typed as duplicate (IoU >= 0.5 with an already matched ground truth of
its class), class (IoU >= 0.5 with a ground truth of another class), localization
(0.1 <= IoU < 0.5 with its class) or background. ``stats`` writes {results}/qualitative.json;
``figure`` draws Figure 3: three images with both rules, and two clusters whose ASA box
crosses IoU 0.75 with the ground truth while the WBF box stays below it (members, WBF box,
ASA box and ground truth; the members come from the exported views). The IoUs are printed with
three decimals, so that a WBF box just below 0.75 is not shown as 0.75.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.clusters import clusters_for_image  # noqa: E402
from asa.config import Experiments  # noqa: E402
from asa.obb import fast_iou, read_obb_file  # noqa: E402
from asa.views import FAMILIES  # noqa: E402

CELLS = ["fair1m2_yolo11m:d2", "fair1m2_yolo11m:c4", "fair1m2_yolo11m:d4", "fair1m2_yolo11m:ms", "dota_orcnn:ms"]
RULES = {"wbf": "wbf_orig", "asa": "asa"}
# (detector, family, image, crop x0, y0, side in pixels, column title) of Figure 3
EXAMPLES = [("fair1m2_yolo11m", "c4", "1101", 140, 0, 600, "(a) FAIR1M-2.0, YOLO11m, C4"),
            ("fair1m2_yolo11m", "c4", "6143", 230, 40, 560, "(b) FAIR1M-2.0, YOLO11m, C4"),
            ("dota_orcnn", "ms", "P0411__1024__0___62", 170, 200, 420, "(c) DOTA-v1.0, O-RCNN, MS")]
ZOOM_CELL, ZOOMS, ZOOM_SCAN = ("fair1m2_yolo11m", "c4"), [("1102", 15), ("2493", 14)], 500


def match(dets, gts):
    """dets sorted by score -> per det (is_tp, gt index), and the set of matched ground truths."""
    used, out = set(), []
    for d in dets:
        best, bi = 0.5, None
        for j, g in enumerate(gts):
            if j not in used and g["cls"] == d["cls"]:
                iou = fast_iou(d, g)
                if iou >= best:
                    best, bi = iou, j
        if bi is not None:
            used.add(bi)
        out.append((bi is not None, bi))
    return out, used


def fp_type(d, gts, used):
    if any(fast_iou(d, g) >= 0.5 for j, g in enumerate(gts) if g["cls"] == d["cls"] and j in used):
        return "duplicate"
    if max([fast_iou(d, g) for g in gts if g["cls"] != d["cls"]], default=0.0) >= 0.5:
        return "class"
    if max([fast_iou(d, g) for g in gts if g["cls"] == d["cls"]], default=0.0) >= 0.1:
        return "localization"
    return "background"


def predictions(preds, det, fam, rule, name):
    return sorted(read_obb_file(Path(preds) / det / fam / RULES[rule] / name, True), key=lambda b: -b["conf"])[:300]


def cell_stats(ex, preds, det, fam):
    labels = sorted(ex.labels(det, "val").glob("*.txt"))
    gts = {p.name: read_obb_file(p, is_pred=False) for p in labels}
    n_gt = sum(len(v) for v in gts.values())
    kept_all = {r: {p.name: predictions(preds, det, fam, r, p.name) for p in labels} for r in RULES}
    thr = {}
    for r in RULES:
        scores = sorted((b["conf"] for v in kept_all[r].values() for b in v), reverse=True)
        thr[r] = scores[min(n_gt, len(scores)) - 1]
    totals = {r: {"tp": 0, "fp": 0, "duplicate": 0, "class": 0, "localization": 0, "background": 0} for r in RULES}
    for p in labels:
        for r in RULES:
            kept = [b for b in kept_all[r][p.name] if b["conf"] >= thr[r]]
            res, used = match(kept, gts[p.name])
            for b, (tp, _) in zip(kept, res):
                totals[r]["tp" if tp else "fp"] += 1
                if not tp:
                    totals[r][fp_type(b, gts[p.name], used)] += 1
    return {"detector": det, "family": fam, "n_gt": n_gt, "thresholds": thr, "totals": totals}


def box_candidates(ex, preds, det, fam):
    dirs, views = ex.view_dirs(det, "val", fam), FAMILIES[fam]
    out = []
    for p in sorted(ex.labels(det, "val").glob("*.txt"))[:ZOOM_SCAN]:
        gts = read_obb_file(p, is_pred=False)
        asa = read_obb_file(Path(preds) / det / fam / RULES["asa"] / p.name, True)
        for box, members in clusters_for_image(p.name, dirs, views):
            same = [(fast_iou(box, g), j) for j, g in enumerate(gts) if g["cls"] == box["cls"]]
            if not same or max(same)[0] < 0.5 or len(members) < 3:
                continue
            iou_w, j = max(same)
            m = sorted((fast_iou(box, b), i) for i, b in enumerate(asa) if b["cls"] == box["cls"])
            if not m or m[-1][0] < 0.6 or (len(m) > 1 and m[-2][0] > 0.3):
                continue
            iou_a = fast_iou(asa[m[-1][1]], gts[j])
            if iou_a >= 0.75 > iou_w and iou_a - iou_w >= 0.08:
                out.append({"image": p.stem, "cls": box["cls"], "iou_wbf": round(iou_w, 3), "iou_asa": round(iou_a, 3),
                            "gt": gts[j]["poly"], "wbf": box["poly"], "asa": asa[m[-1][1]]["poly"],
                            "members": [mm["poly"] for mm in members]})
    return out


def figure(ex, preds, stats, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    from PIL import Image

    green, verm, yel, sky, pink, white = "#009E73", "#D55E00", "#F0E442", "#56B4E9", "#CC79A7", "#FFFFFF"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 7, "pdf.fonttype": 42, "mathtext.fontset": "stix"})

    def px(poly, w, h, x0=0.0, y0=0.0):
        return np.array([[x * w - x0, y * h - y0] for x, y in poly])

    def image(det, stem):
        return Image.open(next(ex.images(det, "val").glob(stem + ".*"))).convert("RGB")

    def label(ax, text):
        ax.text(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top", color="white", fontsize=6.5,
                bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.6))

    def detections(ax, det, fam, stem, x0, y0, side, rule, thr):
        img = image(det, stem)
        w, h = img.size
        ax.imshow(np.asarray(img.crop((x0, y0, x0 + side, y0 + side))), extent=(0, side, side, 0))
        gts = read_obb_file(ex.labels(det, "val") / f"{stem}.txt", is_pred=False)
        kept = [b for b in predictions(preds, det, fam, rule, f"{stem}.txt") if b["conf"] >= thr]
        res, used = match(kept, gts)

        def inside(poly):
            c = px(poly, w, h).mean(0)
            return x0 <= c[0] < x0 + side and y0 <= c[1] < y0 + side

        n = {"tp": 0, "fp": 0, "missed": 0}
        for j, g in enumerate(gts):
            if j not in used and inside(g["poly"]):
                ax.add_patch(Polygon(px(g["poly"], w, h, x0, y0), closed=True, fill=False, ec=yel, lw=0.9,
                                     ls=(0, (2, 1.2))))
                n["missed"] += 1
        for b, (tp, _) in zip(kept, res):
            if inside(b["poly"]):
                ax.add_patch(Polygon(px(b["poly"], w, h, x0, y0), closed=True, fill=False, ec=green if tp else verm,
                                     lw=0.9))
                n["tp" if tp else "fp"] += 1
        ax.set_xlim(0, side)
        ax.set_ylim(side, 0)
        label(ax, f"TP {n['tp']}  FP {n['fp']}  missed {n['missed']}")

    def zoom(ax, det, c):
        img = image(det, c["image"])
        w, h = img.size
        gt = px(c["gt"], w, h)
        allp = np.concatenate([gt] + [px(m, w, h) for m in c["members"]])
        cx, cy = gt.mean(0)
        half = max(16.0, 0.62 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])))
        x0 = min(max(cx - half, 0.0), max(w - 2 * half, 0.0))
        y0 = min(max(cy - half, 0.0), max(h - 2 * half, 0.0))
        ax.imshow(np.asarray(img), extent=(0, w, h, 0), interpolation="bicubic")
        for m in c["members"]:
            ax.add_patch(Polygon(px(m, w, h), closed=True, fill=False, ec=white, lw=0.6, alpha=0.9))
        ax.add_patch(Polygon(gt, closed=True, fill=False, ec=yel, lw=1.1))
        ax.add_patch(Polygon(px(c["wbf"], w, h), closed=True, fill=False, ec=sky, lw=1.1, ls=(0, (2.2, 1.2))))
        ax.add_patch(Polygon(px(c["asa"], w, h), closed=True, fill=False, ec=pink, lw=1.1))
        ax.set_xlim(x0, x0 + 2 * half)
        ax.set_ylim(y0 + 2 * half, y0)
        label(ax, f"IoU {c['iou_wbf']:.3f} → {c['iou_asa']:.3f}")

    cands = box_candidates(ex, preds, *ZOOM_CELL)
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.72), gridspec_kw=dict(wspace=0.04, hspace=0.06))
    for col, (det, fam, stem, x0, y0, side, title) in enumerate(EXAMPLES):
        thr = stats[f"{det}:{fam}"]["thresholds"]
        for row, r in enumerate(("wbf", "asa")):
            detections(axes[row, col], det, fam, stem, x0, y0, side, r, thr[r])
        axes[0, col].set_title(title, fontsize=7, pad=3)
    for row, (stem, cls) in enumerate(ZOOMS):
        zoom(axes[row, 3], ZOOM_CELL[0], next(c for c in cands if c["image"] == stem and c["cls"] == cls))
    axes[0, 3].set_title("(d) Learned box weights", fontsize=7, pad=3)
    axes[0, 0].set_ylabel("WBF, original score", fontsize=7)
    axes[1, 0].set_ylabel("ASA", fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    handles = [Line2D([], [], color=green, lw=1.2, label="true positive"),
               Line2D([], [], color=verm, lw=1.2, label="false positive"),
               Line2D([], [], color=yel, lw=1.2, ls=(0, (2, 1.2)), label="missed object"),
               Line2D([], [], color="0.55", lw=0.8, label="(d) cluster members"),
               Line2D([], [], color=sky, lw=1.2, ls=(0, (2.2, 1.2)), label="WBF box"),
               Line2D([], [], color=pink, lw=1.2, label="ASA box"),
               Line2D([], [], color=yel, lw=1.2, label="ground truth")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.94, bottom=0.075)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01, dpi=300)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stats", "figure"])
    ap.add_argument("--config", default="configs/experiments.json")
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--results", default="results")
    ap.add_argument("--cells", nargs="+", default=CELLS, help="detector:family")
    ap.add_argument("--out", default="figure_qualitative.pdf")
    a = ap.parse_args()
    ex = Experiments(a.config)
    path = Path(a.results) / "qualitative.json"
    if a.what == "stats":
        res = {}
        for cell in a.cells:
            det, fam = cell.split(":")
            r = res[cell] = cell_stats(ex, a.preds, det, fam)
            t = r["totals"]
            print(f"{cell}: true positives WBF {t['wbf']['tp']}, ASA {t['asa']['tp']} of {r['n_gt']}; false positives "
                  + ", ".join(f"{k} {t['wbf'][k]} -> {t['asa'][k]}" for k in ("class", "background", "localization",
                                                                              "duplicate")), flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1))
    else:
        figure(ex, a.preds, json.loads(path.read_text()), a.out)


if __name__ == "__main__":
    main()
