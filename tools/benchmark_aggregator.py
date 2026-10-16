"""Per-image CPU cost of ASA on top of rotated WBF, single thread (Table 11).

    python tools/benchmark_aggregator.py --detector dota_yolo11s --family c4 --n 100

On the first ``--n`` validation images it times
    wbf       clustering and fused boxes (asa.fusion.classwise_wbf)
    features  clustering, cluster and member descriptors (asa.clusters.build_members), minus
              the time of reading the prediction files, which wbf does not include either
    tensors   building the network input
    network   the forward pass (untrained weights: the cost does not depend on them)
The prediction files are read once beforehand, so all timings are warm. Image sizes are
passed in (they are known at inference) and no ground truth is read. Writes
``results/latency/aggregator_<detector>_<family>.json`` unless ``--out`` is given.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.clusters import build_members  # noqa: E402
from asa.config import Experiments  # noqa: E402
from asa.fusion import classwise_wbf  # noqa: E402
from asa.model import make_net, predict, to_tensors  # noqa: E402
from asa.obb import image_aspects, read_obb_file  # noqa: E402
from asa.views import FAMILIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.json")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)

    ex = Experiments(a.config)
    views = FAMILIES[a.family]
    dirs = ex.view_dirs(a.detector, "val", a.family)
    names = [p.name for p in sorted(ex.labels(a.detector, "val").glob("*.txt"))[:a.n]]
    aspects = image_aspects(ex.images(a.detector, "val"), [Path(n).stem for n in names])

    def read():
        return {n: [read_obb_file(d / n, is_pred=True) for d in dirs] for n in names}

    read()  # warm the file cache
    t0 = time.perf_counter()
    preds = read()
    t_io = time.perf_counter() - t0
    t0 = time.perf_counter()
    for per_view in preds.values():
        classwise_wbf([b for v in per_view for b in v], 0.55, "max")
    t_wbf = time.perf_counter() - t0
    t0 = time.perf_counter()
    z = build_members(None, None, dirs, views, aspects=aspects, with_labels=False, names=names)
    t_feat = time.perf_counter() - t0 - t_io  # build_members reads the files itself
    t0 = time.perf_counter()
    T = to_tensors(z, len(views))
    net = make_net(T["mfeat"].shape[-1], T["xc"].shape[-1], ex.num_classes(a.detector))
    t_tensors = time.perf_counter() - t0
    t0 = time.perf_counter()
    predict(net, T)
    t_net = time.perf_counter() - t0

    n = len(names)
    res = {"detector": a.detector, "family": a.family, "images": n, "clusters": int(len(z["y"])),
           "ms_per_image": {"io": 1e3 * t_io / n, "wbf": 1e3 * t_wbf / n, "features": 1e3 * t_feat / n,
                            "tensors": 1e3 * t_tensors / n, "network": 1e3 * t_net / n}}
    out = Path(a.out or f"results/latency/aggregator_{a.detector}_{a.family}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
