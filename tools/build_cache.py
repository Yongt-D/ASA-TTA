"""Build the cluster caches of every (detector, family, split): clusters, descriptors, members.

    python tools/build_cache.py --detectors hrsc_yolo11s --families c4 --jobs 8

Writes {cache}/{detector}_{family}_{split}.npz; existing caches are kept.
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.clusters import build_members  # noqa: E402
from asa.config import Experiments  # noqa: E402
from asa.views import FAMILIES, check_backmapping  # noqa: E402

FAMS = ["d2", "c4", "d4", "ms"]


def build(job):
    config, det, fam, split, cache_dir = job
    ex = Experiments(config)
    out = Path(cache_dir) / f"{det}_{fam}_{split}.npz"
    if out.exists():
        return f"{out.name}: exists"
    t0 = time.time()
    dirs = ex.view_dirs(det, split, fam)
    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f"missing view directory {d}")
        check_backmapping(d)
    z = build_members(ex.labels(det, split), ex.images(det, split), dirs, FAMILIES[fam])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **z)
    tmp.replace(out)
    return f"{out.name}: {len(z['y'])} clusters, {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.json")
    ap.add_argument("--detectors", nargs="+")
    ap.add_argument("--families", nargs="+", default=FAMS, choices=list(FAMILIES))
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--cache", default="caches")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    dets = a.detectors or list(Experiments(a.config).detectors)
    jobs = [(a.config, d, f, s, a.cache) for d in dets for f in a.families for s in a.splits]
    with ProcessPoolExecutor(a.jobs) as pool:
        for msg in pool.map(build, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
