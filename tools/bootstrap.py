"""Paired bootstrap intervals, with the resamples of a dataset shared by all of its cells.

Contrasts between rules (Table 5), mean over the cells of each dataset and per cell:

    python tools/bootstrap.py contrasts --pairs asa:wbf_orig asa:obbstacking_wbfbox

Each fixed-rule family against the single view r0, per detector (Table 1; run
``tools/fuse.py --rule wbf_max --families flip`` first for the Flip family):

    python tools/bootstrap.py vs-r0 --rule wbf_max --families flip d2 c4 d4 ms

DOTA resamples source scenes (``scene_pattern`` in the config), the other datasets images.
``contrasts`` draws all resamples from one random sequence (seed 3086); ``vs-r0`` seeds
resample b with 3086 + b, as the evaluation of Table 1 did. Results go to
{results}/bootstrap_contrasts.json and {results}/bootstrap_vs_r0.json.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.bootstrap import ap50, match_source, percentile_ci, resample_weights  # noqa: E402
from asa.config import Experiments  # noqa: E402

FAMS = ["d2", "c4", "d4", "ms"]


def cell_job(args):
    """AP50 of every source of one cell on the full set and on every resample."""
    label_dir, num_classes, sources, W = args
    full = np.ones(W.shape[1], np.int64)
    out = {}
    for name, pred_dir in sources.items():
        m = match_source(label_dir, pred_dir, num_classes)
        out[name] = {"full": 100 * ap50(m, full, num_classes),
                     "boot": [100 * ap50(m, w, num_classes) for w in W]}
    return out


def run(a, ex, dataset, cells, W):
    """cells: {key: {source: pred_dir}} -> {key: cell_job result}."""
    labels = ex.data_root / ex.datasets[dataset]["root"] / "labels" / "val"
    ncls = int(ex.datasets[dataset]["num_classes"])
    keys = list(cells)
    with ProcessPoolExecutor(a.workers) as pool:
        res = list(pool.map(cell_job, [(labels, ncls, cells[k], W) for k in keys]))
    return dict(zip(keys, res))


def contrast(c, x, y):
    d = np.asarray(c[x]["boot"]) - np.asarray(c[y]["boot"])
    return c[x]["full"] - c[y]["full"], d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["contrasts", "vs-r0"])
    ap.add_argument("--config", default="configs/experiments.json")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--detectors", nargs="+", help="restrict to these detectors")
    ap.add_argument("--families", nargs="+")
    ap.add_argument("--pairs", nargs="+", default=["asa:wbf_orig", "asa:obbstacking_wbfbox", "asa_score:wbf_orig"])
    ap.add_argument("--rule", default="wbf_max", help="vs-r0: the fused rule compared with r0")
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--results", default="results")
    ap.add_argument("--repeats", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3086)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ex = Experiments(a.config)
    fams = a.families or (FAMS if a.mode == "contrasts" else ["flip"] + FAMS)
    out_path = Path(a.results) / ("bootstrap_contrasts.json" if a.mode == "contrasts" else "bootstrap_vs_r0.json")
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    for ds in a.datasets or list(ex.datasets):
        t0 = time.time()
        dets = [d for d in ex.detectors_of(ds) if not a.detectors or d in a.detectors]
        if not dets:
            continue
        labels = ex.data_root / ex.datasets[ds]["root"] / "labels" / "val"
        W, n_units = resample_weights(labels, a.repeats, a.seed, ex.datasets[ds].get("scene_pattern"),
                                      per_repeat_seed=(a.mode == "vs-r0"))
        unit = "scene" if ex.datasets[ds].get("scene_pattern") else "image"
        if a.mode == "contrasts":
            names = sorted({n for p in a.pairs for n in p.split(":")})
            cells = {f"{d}|{f}": {n: Path(a.preds) / d / f / n for n in names} for d in dets for f in fams}
            res = run(a, ex, ds, cells, W)
            summary = {"units": n_units, "unit": unit, "cells": len(cells), "repeats": a.repeats}
            for pair in a.pairs:
                x, y = pair.split(":")
                per, fulls, boots = {}, [], []
                for k, c in res.items():
                    full, d = contrast(c, x, y)
                    per[k] = {"diff": full, "ci95": percentile_ci(d)}
                    fulls.append(full)
                    boots.append(d)
                mean_boot = np.nanmean(np.stack(boots), axis=0)
                summary[f"{x}_vs_{y}"] = {
                    "mean_diff": float(np.mean(fulls)), "ci95": percentile_ci(mean_boot),
                    "cells_ci_above_0": sum(v["ci95"][0] > 0 for v in per.values()),
                    "cells_ci_below_0": sum(v["ci95"][1] < 0 for v in per.values()), "per_cell": per}
                s = summary[f"{x}_vs_{y}"]
                print(f"{ds:5s} {x} - {y}: {s['mean_diff']:+.2f} AP50, 95% CI [{s['ci95'][0]:+.2f}, "
                      f"{s['ci95'][1]:+.2f}], cells with CI > 0: {s['cells_ci_above_0']}/{len(per)}, "
                      f"< 0: {s['cells_ci_below_0']}", flush=True)
        else:
            cells = {d: {"r0": ex.view_dir(d, "val", "r0"), **{f: Path(a.preds) / d / f / a.rule for f in fams}}
                     for d in dets}
            res = run(a, ex, ds, cells, W)
            summary = {"units": n_units, "unit": unit, "rule": a.rule, "repeats": a.repeats, "detectors": {}}
            for d, c in res.items():
                summary["detectors"][d] = {}
                for f in fams:
                    full, diff = contrast(c, f, "r0")
                    summary["detectors"][d][f] = {"diff": full, "ci95": percentile_ci(diff)}
                print(ds, d, " ".join(f"{f} {summary['detectors'][d][f]['diff']:+.2f}" for f in fams), flush=True)
        report[ds] = summary
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=1))
        print(f"{ds}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
