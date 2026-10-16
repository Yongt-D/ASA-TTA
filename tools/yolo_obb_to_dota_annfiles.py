"""Convert YOLO-OBB labels to the DOTA annotation files MMRotate reads, and check the round trip.

    python tools/yolo_obb_to_dota_annfiles.py --yaml data/HRSC2016_MS_yolo_obb/hrsc2016_ms_obb.yaml \
        --split train --out data/mmrotate/hrsc/annfiles/train --link-images

ours  ``cls x1 y1 ... x4 y4``               corners normalised by the image size
DOTA  ``x1 y1 ... x4 y4 name difficulty``   corners in pixels, difficulty 0

Class names come from the dataset yaml. MMRotate splits annotation lines on whitespace, so
spaces in names become hyphens; the names are written to ``classes.json`` (our index -> name)
next to ``--out``, which the MMRotate configs list and the exporter matches by name. Every
file is read back and compared with its source. ``--min-side N`` drops boxes whose shorter
side is under N pixels: a zero-width box turns the rotated-box regression target into NaN in
MMRotate (DIOR-R train has one, image 04137; we used ``--min-side 2`` there).
``--link-images`` links ``images/<split>`` beside ``annfiles`` to the dataset's images.
"""
import argparse
import json
import math
from pathlib import Path

import yaml
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def class_tokens(yaml_path):
    names = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    return {int(k): str(v).strip().replace(" ", "-") for k, v in names.items()}


def image_size(img_dir, stem):
    for ext in IMG_EXT:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            with Image.open(p) as im:
                return im.size
    raise SystemExit(f"no image for {stem} under {img_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="dataset yaml (path, names)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", help="dataset root (default: the yaml's path)")
    ap.add_argument("--min-side", type=float, default=0.0)
    ap.add_argument("--link-images", action="store_true")
    a = ap.parse_args()
    root = Path(a.root or yaml.safe_load(Path(a.yaml).read_text(encoding="utf-8"))["path"])
    labels, images = root / "labels" / a.split, root / "images" / a.split
    tokens = class_tokens(a.yaml)
    if len(set(tokens.values())) != len(tokens):
        raise SystemExit("class names are not unique after replacing spaces")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out.parent / "classes.json").write_text(json.dumps(tokens, indent=2))
    if a.link_images:
        link = out.parent.parent / "images" / a.split
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(images.resolve(), target_is_directory=True)

    counts = {t: 0 for t in tokens.values()}
    worst, files, dropped = 0.0, 0, []
    for lab in sorted(labels.glob("*.txt")):
        w, h = image_size(images, lab.stem)
        src, lines = [], []
        for line in lab.read_text().splitlines():
            f = line.split()
            if not f:
                continue
            if len(f) != 9:
                raise SystemExit(f"{lab}: expected 9 fields, got {line!r}")
            cls, xy = int(f[0]), [float(v) for v in f[1:]]
            px = [v * (w if i % 2 == 0 else h) for i, v in enumerate(xy)]
            pts = [(px[i], px[i + 1]) for i in range(0, 8, 2)]
            edge = [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
            if min((edge[0] + edge[2]) / 2, (edge[1] + edge[3]) / 2) < a.min_side:
                dropped.append(f"{lab.name}:{tokens[cls]}")
                continue
            lines.append(" ".join(f"{v:.4f}" for v in px) + f" {tokens[cls]} 0")
            src.append((tokens[cls], xy))
            counts[tokens[cls]] += 1
        dst = out / lab.name
        dst.write_text("\n".join(lines) + ("\n" if lines else ""))
        back = [(f[8], [float(v) / (w if i % 2 == 0 else h) for i, v in enumerate(f[:8])])
                for f in (line.split() for line in dst.read_text().splitlines())]
        if [b[0] for b in back] != [s[0] for s in src]:
            raise SystemExit(f"{dst}: class order changed in the round trip")
        for (_, xy_a), (_, xy_b) in zip(src, back):
            worst = max(worst, max(abs(p - q) * max(w, h) for p, q in zip(xy_a, xy_b)))
        files += 1
    if worst > 0.001:
        raise SystemExit(f"round trip off by {worst:.4f} px")
    print(json.dumps({"files": files, "instances": sum(counts.values()), "max_roundtrip_px": worst,
                      "dropped": dropped, "per_class": counts}, indent=1))


if __name__ == "__main__":
    main()
