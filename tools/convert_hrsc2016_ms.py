"""Convert HRSC2016-MS to the YOLO-OBB layout (single class ``ship``).

    python tools/convert_hrsc2016_ms.py --root data/HRSC2016_MS/raw --out-root data/HRSC2016_MS_yolo_obb

``--root`` holds ``AllImages/*.bmp``, ``Annotations/*.xml`` and ``ImageSets/{train,val}.txt``
(or ``ImageSets/source/``). Each rotated box (cx, cy, w, h, angle) becomes its four corners,
clipped to the image and normalised by its size; images without boxes are skipped. Writes
``labels/{train,val}``, ``images/{train,val}`` (symbolic links, or copies where links are not
allowed) and ``hrsc2016_ms_obb.yaml``. Our splits: 603 training and 454 validation images.
"""
import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def read_split(root, split):
    for path in (root / "ImageSets" / f"{split}.txt", root / "ImageSets" / "source" / f"{split}.txt"):
        if path.exists():
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    raise FileNotFoundError(f"no split file for {split} under {root / 'ImageSets'}")


def rect_points(cx, cy, width, height, angle):
    ca, sa = math.cos(angle), math.sin(angle)
    dx, dy = width / 2.0, height / 2.0
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in ((-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy))]


def parse_xml(path):
    root = ET.parse(path).getroot()
    width = int(float(root.findtext("size/width")))
    height = int(float(root.findtext("size/height")))
    rows = []
    for obj in root.findall("object"):
        box = obj.find("robndbox")
        if box is None:
            continue
        cx, cy, w, h, angle = (float(box.findtext(k)) for k in ("cx", "cy", "w", "h", "angle"))
        rows.append([(max(0.0, min(width - 1.0, x)) / width, max(0.0, min(height - 1.0, y)) / height)
                     for x, y in rect_points(cx, cy, w, h, angle)])
    return rows


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--skip-images", action="store_true")
    a = ap.parse_args()
    root, out_root = Path(a.root), Path(a.out_root)

    stats = {}
    for split, name in (("train", a.train_split), ("val", a.val_split)):
        images = objects = 0
        for stem in read_split(root, name):
            xml_path, image_path = root / "Annotations" / f"{stem}.xml", root / "AllImages" / f"{stem}.bmp"
            if not xml_path.exists() or not image_path.exists():
                continue
            rows = parse_xml(xml_path)
            if not rows:
                continue
            label = out_root / "labels" / split / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("".join("0 " + " ".join(f"{v:.8f}" for pt in pts for v in pt) + "\n" for pts in rows))
            if not a.skip_images:
                link_or_copy(image_path, out_root / "images" / split / image_path.name)
            images += 1
            objects += len(rows)
        stats[split] = {"images": images, "objects": objects}
    (out_root / "hrsc2016_ms_obb.yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ship\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
