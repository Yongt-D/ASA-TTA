"""Fixed fusion rules on the views of a family: rotated WBF, rotated NMS, Soft-NMS.

    python tools/fuse.py --rule nms softnms --families c4        # Table 8
    python tools/fuse.py --rule wbf_max --families flip          # the Flip family of Table 1

Writes {preds}/{detector}/{family}/{rule}/ and adds the metrics (percent) to
{results}/{detector}/{family}.json. WBF for the families d2/c4/d4/ms is also produced by
tools/run_cell.py from the cluster caches (identical boxes and scores).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.config import Experiments  # noqa: E402
from asa.evaluate import evaluate  # noqa: E402
from asa.fusion import RULES, pooled_predictions  # noqa: E402
from asa.obb import write_obb_file  # noqa: E402
from asa.views import FAMILIES, check_backmapping  # noqa: E402


def fuse_image(args):
    name, view_dirs, rule, iou_thr, max_dets, out_dir = args
    fused = RULES[rule](pooled_predictions(name, view_dirs), iou_thr)[:max_dets]
    write_obb_file(Path(out_dir) / name, fused)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.json")
    ap.add_argument("--detectors", nargs="+")
    ap.add_argument("--families", nargs="+", required=True, choices=list(FAMILIES))
    ap.add_argument("--rule", nargs="+", required=True, choices=list(RULES))
    ap.add_argument("--iou-thr", type=float, default=0.55)
    ap.add_argument("--max-dets", type=int, default=1000)
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--results", default="results")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ex = Experiments(a.config)
    for det in a.detectors or list(ex.detectors):
        labels = ex.labels(det, "val")
        names = [p.name for p in sorted(labels.glob("*.txt"))]
        for fam in a.families:
            dirs = ex.view_dirs(det, "val", fam)
            for d in dirs:
                check_backmapping(d)
            path = Path(a.results) / det / f"{fam}.json"
            res = json.loads(path.read_text()) if path.exists() else {}
            for rule in a.rule:
                out = Path(a.preds) / det / fam / rule
                with ProcessPoolExecutor(a.workers) as pool:
                    list(pool.map(fuse_image, [(n, dirs, rule, a.iou_thr, a.max_dets, out) for n in names], chunksize=16))
                r = evaluate(labels, out, ex.num_classes(det), 300, a.workers)
                res[rule] = {k: 100 * r[k] for k in ("mAP50", "AP75", "mAP50_95")}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(res, indent=1))
                print(f"{det} {fam} {rule}: AP50 {res[rule]['mAP50']:.2f} AP75 {res[rule]['AP75']:.2f}", flush=True)


if __name__ == "__main__":
    main()
