"""Fit and evaluate every aggregation rule on one or more cells (detector x TTA family).

    python tools/run_cell.py --detectors hrsc_yolo11s --families c4 [--ablations]

For each cell the rules score the same rotated-WBF clusters of the evaluation split:

    wbf_max             WBF, max score
    wbf_orig            WBF, original score p0 (the prior of ASA)
    gbdt                GBDT re-scoring, WBF box
    obbstacking         OBBStacking with its own box fusion
    obbstacking_wbfbox  OBBStacking score, WBF box
    asa_score           ASA score, WBF box
    asa                 ASA score and learned box weights
  with --ablations (Table 9):
    asa_no_prior        score predicted freely (no WBF prior, no penalty)
    asa_no_penalty      WBF prior without the penalty on the residual
    asa_no_members      cluster descriptor only (member descriptors zeroed)
    wbf_orig_gaussbox   original WBF score, Gaussian average of the members as box

Predictions go to {preds}/{detector}/{family}/{rule}/, metrics (percent) to
{results}/{detector}/{family}.json; the single view r0 to {results}/{detector}/r0.json.
Several cells are run one subprocess each, which returns all memory after every cell.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.baselines import gbdt_scores, obbstacking  # noqa: E402
from asa.clusters import boxes_from, write_rescored  # noqa: E402
from asa.config import Experiments  # noqa: E402
from asa.evaluate import evaluate  # noqa: E402
from asa.model import DEFAULTS, fit, gauss_average_polys, predict, to_tensors, wbf_avg_score  # noqa: E402
from asa.views import FAMILIES  # noqa: E402

FAMS = ["d2", "c4", "d4", "ms"]
ABLATIONS = {"asa_no_prior": dict(anchor=False, delta_l2=0.0), "asa_no_penalty": dict(delta_l2=0.0),
             "asa_no_members": dict(members=False)}


def pct(r):
    return {k: 100 * r[k] for k in ("mAP50", "AP75", "mAP50_95")}


def run_cell(a, det, fam):
    import torch
    torch.set_num_threads(a.threads)
    ex = Experiments(a.config)
    nv, ncls = len(FAMILIES[fam]), ex.num_classes(det)
    labels = ex.labels(det, "val")
    rdir = Path(a.results) / det
    rdir.mkdir(parents=True, exist_ok=True)
    r0_path = rdir / "r0.json"
    if not r0_path.exists():
        r0_path.write_text(json.dumps(pct(evaluate(labels, ex.view_dir(det, "val", "r0"), ncls, 300, a.workers)), indent=1))
    path = rdir / f"{fam}.json"
    res = json.loads(path.read_text()) if path.exists() and not a.force else {}
    tr = dict(np.load(Path(a.cache) / f"{det}_{fam}_train.npz"))
    te = dict(np.load(Path(a.cache) / f"{det}_{fam}_val.npz"))
    wbf_boxes = boxes_from(te, te["fpoly"])
    t0 = time.time()

    def emit(name, scores, polys=None):
        out = Path(a.preds) / det / fam / name
        write_rescored(out, wbf_boxes if polys is None else boxes_from(te, polys), scores, labels)
        res[name] = pct(evaluate(labels, out, ncls, 300, a.workers))
        path.write_text(json.dumps(res, indent=1))
        print(f"{det} {fam} {name:20s} AP50 {res[name]['mAP50']:.2f}  AP75 {res[name]['AP75']:.2f}  "
              f"AP50:95 {res[name]['mAP50_95']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    def todo(*names):
        return any(n not in res for n in names)

    if todo("wbf_max"):
        emit("wbf_max", te["X"][:, 9])
    if todo("wbf_orig"):
        emit("wbf_orig", wbf_avg_score(te["X"], nv))
    if todo("gbdt"):
        emit("gbdt", gbdt_scores(tr, te))
    if todo("obbstacking", "obbstacking_wbfbox"):
        sc, pp = obbstacking(tr, te, nv)
        emit("obbstacking", sc, pp)
        emit("obbstacking_wbfbox", sc)
    variants = {"asa": {}} | (ABLATIONS if a.ablations else {})
    if todo("asa_score", *variants) or (a.ablations and todo("wbf_orig_gaussbox")):
        Ttr, Tte = to_tensors(tr, nv), to_tensors(te, nv)
        for name, over in variants.items():
            if not todo(name) and (name != "asa" or not todo("asa_score")):
                continue
            print(f"{det} {fam}: fitting {name} on {len(tr['y'])} training clusters", flush=True)
            net = fit(Ttr, nv, ncls, **(DEFAULTS | over))
            s, polys = predict(net, Tte)
            emit(name, s, polys)
            if name == "asa":
                emit("asa_score", s)
        if a.ablations and todo("wbf_orig_gaussbox"):
            emit("wbf_orig_gaussbox", wbf_avg_score(te["X"], nv), gauss_average_polys(Tte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.json")
    ap.add_argument("--detectors", nargs="+")
    ap.add_argument("--families", nargs="+", default=FAMS, choices=FAMS)
    ap.add_argument("--cache", default="caches")
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--results", default="results")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    ap.add_argument("--workers", type=int, default=8, help="evaluation processes")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute rules that already have results")
    ap.add_argument("--cell", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.cell:
        det, fam = a.cell.split(":")
        return run_cell(a, det, fam)
    dets = a.detectors or list(Experiments(a.config).detectors)
    cells = [(d, f) for d in dets for f in a.families]
    if len(cells) == 1:
        return run_cell(a, *cells[0])
    for det, fam in cells:
        subprocess.run([sys.executable, __file__, *sys.argv[1:], "--cell", f"{det}:{fam}"], check=True)


if __name__ == "__main__":
    main()
