"""Train ASA on a detector's TTA predictions for its training images, and save the model.

    python tools/train_asa.py --family c4 --labels data/mydata/labels/train \
        --images data/mydata/images/train --num-classes 15 --out models/mydet_c4.pt \
        --views views/mydet/train/base/r0 views/mydet/train/base/r90 \
                views/mydet/train/base/r180 views/mydet/train/base/r270

``--views`` gives one prediction directory per view of the family, in the family's order
(``asa.views.FAMILIES``), as written by tools/export_views_*.py. The settings are those of
the paper (``asa.model.DEFAULTS``). Apply the model with tools/apply_asa.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.clusters import build_members  # noqa: E402
from asa.model import DEFAULTS, fit, save, to_tensors  # noqa: E402
from asa.views import FAMILIES, check_backmapping  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--views", nargs="+", required=True, help="one directory per view, in the family's order")
    ap.add_argument("--labels", required=True, help="training labels (YOLO-OBB format)")
    ap.add_argument("--images", required=True, help="training images (for their aspect ratios)")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)

    views = FAMILIES[a.family]
    if len(a.views) != len(views):
        raise SystemExit(f"{a.family} has {len(views)} views ({' '.join(views)}), got {len(a.views)} directories")
    dirs = [Path(d) for d in a.views]
    for d in dirs:
        check_backmapping(d)
    z = build_members(a.labels, a.images, dirs, views)
    T = to_tensors(z, len(views))
    print(f"fitting ASA ({a.family}) on {len(z['y'])} clusters of {len(z['names'])} images", flush=True)
    net = fit(T, len(views), a.num_classes, **DEFAULTS)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save(net, a.out, family=a.family, views=views, num_classes=a.num_classes, d_member=int(T["mfeat"].shape[-1]),
         d_cluster=int(T["xc"].shape[-1]), box_mode=DEFAULTS["box_mode"], iou_thr=0.55)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
