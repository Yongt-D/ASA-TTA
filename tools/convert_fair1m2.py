"""Convert FAIR1M-2.0 (37 classes) to the YOLO-OBB layout, with our train/val split.

    python tools/convert_fair1m2.py --images data/FAIR1M2_raw/images --xmls data/FAIR1M2_raw/labelXml \
        --out-root data/FAIR1M2_yolo_obb_rgb --splits splits

The official validation labels are not public, so the official training set is split:
images with at least one box are shuffled with seed 42 and 20% go to validation (6,630
training and 1,657 validation images). ``--splits`` reads the shipped lists
``fair1m2_seed42_{train,val}.txt`` instead and checks that every listed image was found.
Corners are clipped to the image and normalised by its size; classes are sorted by name.
Images are re-encoded as RGB JPEG (Pillow defaults), as the detectors were trained on.
"""
import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

IMAGE_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png")


def find_image(image_dir, stem):
    for ext in IMAGE_EXTS:
        path = Path(image_dir) / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def parse_xml(path):
    root = ET.parse(path).getroot()
    width = int(float(root.findtext("size/width")))
    height = int(float(root.findtext("size/height")))
    objects = []
    for obj in root.findall("objects/object"):
        name = obj.findtext("possibleresult/name")
        points = []
        for point in obj.findall("points/point")[:4]:
            x, y = (float(v) for v in point.text.split(","))
            points.append((max(0.0, min(width - 1.0, x)) / width, max(0.0, min(height - 1.0, y)) / height))
        if name and len(points) == 4:
            objects.append((name.strip(), points))
    return objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--xmls", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--splits", help="directory with fair1m2_seed42_{train,val}.txt")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--skip-images", action="store_true")
    a = ap.parse_args()

    parsed, names = {}, set()
    for xml_path in sorted(Path(a.xmls).glob("*.xml")):
        objects = parse_xml(xml_path)
        image_path = find_image(a.images, xml_path.stem)
        if objects and image_path is not None:
            parsed[xml_path.stem] = (image_path, objects)
            names.update(n for n, _ in objects)
    classes = sorted(names)
    class_to_idx = {n: i for i, n in enumerate(classes)}

    if a.splits:
        split_of = {}
        for split in ("train", "val"):
            listed = (Path(a.splits) / f"fair1m2_seed42_{split}.txt").read_text().split()
            missing = [s for s in listed if s not in parsed]
            if missing:
                raise SystemExit(f"{len(missing)} listed {split} images not found, e.g. {missing[:5]}")
            split_of.update({s: split for s in listed})
    else:
        stems = sorted(parsed)
        random.Random(a.seed).shuffle(stems)
        n_val = max(1, int(round(len(stems) * a.val_ratio)))
        split_of = {s: "val" if i < n_val else "train" for i, s in enumerate(stems)}

    out_root = Path(a.out_root)
    stats = {"train": {"images": 0, "objects": 0}, "val": {"images": 0, "objects": 0}}
    for stem, split in split_of.items():
        image_path, objects = parsed[stem]
        label = out_root / "labels" / split / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("".join(f"{class_to_idx[n]} " + " ".join(f"{v:.8f}" for pt in pts for v in pt) + "\n"
                                 for n, pts in objects))
        if not a.skip_images:
            dst = out_root / "images" / split / f"{stem}.{a.image_format}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                with Image.open(image_path) as im:
                    im.convert("RGB").save(dst)
        stats[split]["images"] += 1
        stats[split]["objects"] += len(objects)
    (out_root / "fair1m_obb.yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes)))
    print(json.dumps({"classes": len(classes), **stats}, indent=1))


if __name__ == "__main__":
    main()
