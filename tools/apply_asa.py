"""Aggregate TTA predictions with a trained ASA model; no labels needed.

    python tools/apply_asa.py --model models/mydet_c4.pt --images data/mydata/images/test \
        --out preds/mydet_c4_asa \
        --views views/mydet/test/base/r0 views/mydet/test/base/r90 \
                views/mydet/test/base/r180 views/mydet/test/base/r270

Writes one file per image, ``cls x1 y1 ... x4 y4 score`` with vertices normalised by the
image size, sorted by score, at most ``--max-dets`` boxes (images without predictions get an
empty file). ``--labels`` restricts the images to those with a label file, as in the paper's
evaluation; ``python -m asa.evaluate`` then scores the output.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.clusters import boxes_from, build_members, write_rescored  # noqa: E402
from asa.model import load, predict, to_tensors  # noqa: E402
from asa.obb import IMAGE_SUFFIXES  # noqa: E402
from asa.views import check_backmapping  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--views", nargs="+", required=True, help="one directory per view, in the family's order")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", help="only images with a label file here")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dets", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)

    net, meta = load(a.model)
    views = meta["views"]
    if len(a.views) != len(views):
        raise SystemExit(f"the model aggregates {len(views)} views ({' '.join(views)}), got {len(a.views)} directories")
    dirs = [Path(d) for d in a.views]
    for d in dirs:
        check_backmapping(d)
    stems = sorted(p.stem for p in Path(a.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if a.labels:
        labelled = {p.stem for p in Path(a.labels).glob("*.txt")}
        stems = [s for s in stems if s in labelled]
    names = [f"{s}.txt" for s in stems]
    z = build_members(None, a.images, dirs, views, iou_thr=meta["iou_thr"], with_labels=False, names=names)
    scores, polys = predict(net, to_tensors(z, len(views)))
    write_rescored(a.out, boxes_from(z, polys), scores, None, max_dets=a.max_dets, names=names)
    print(f"{len(names)} images, {len(scores)} clusters -> {a.out}")


if __name__ == "__main__":
    main()
