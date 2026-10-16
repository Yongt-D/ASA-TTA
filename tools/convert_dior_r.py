"""Convert DIOR-R (20 classes) to the YOLO-OBB layout, from the official archives.

    python tools/convert_dior_r.py --download-root data/DIOR-R/download --out-root data/DIOR-R/yolo_obb

``--download-root`` holds ``Annotations.zip``, ``ImageSets.zip`` and ``JPEGImages-trainval.zip``
as distributed. The four corners of every oriented box are clipped to the image and
normalised by its size. Writes ``labels/{train,val}``, ``images/{train,val}`` and
``dior_r_obb.yaml`` (the official class order). Our splits are the official train (5,862
images) and val (5,863 images) splits; the test split is not used.
"""
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

CLASSES = ["airplane", "airport", "baseballfield", "basketballcourt", "bridge", "chimney", "dam",
           "Expressway-Service-area", "Expressway-toll-station", "golffield", "groundtrackfield", "harbor",
           "overpass", "ship", "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"]
CORNERS = [("x_left_top", "y_left_top"), ("x_right_top", "y_right_top"),
           ("x_right_bottom", "y_right_bottom"), ("x_left_bottom", "y_left_bottom")]


def read_split(image_sets_zip, split):
    with zipfile.ZipFile(image_sets_zip) as zf:
        text = zf.read(f"Main/{split}.txt").decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    width = float(root.findtext("size/width"))
    height = float(root.findtext("size/height"))
    rows = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("robndbox")
        if name not in CLASSES or box is None:
            continue
        texts = [(box.findtext(kx), box.findtext(ky)) for kx, ky in CORNERS]
        if any(x is None or y is None for x, y in texts):
            continue
        rows.append((CLASSES.index(name), [(max(0.0, min(width - 1.0, float(x))) / width,
                                            max(0.0, min(height - 1.0, float(y))) / height) for x, y in texts]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--skip-images", action="store_true")
    a = ap.parse_args()
    src, out_root = Path(a.download_root), Path(a.out_root)
    stats = {}
    with zipfile.ZipFile(src / "Annotations.zip") as ann, zipfile.ZipFile(src / "JPEGImages-trainval.zip") as imgs:
        for split in ("train", "val"):
            images = objects = 0
            for stem in read_split(src / "ImageSets.zip", split):
                try:
                    rows = parse_xml(ann.read(f"Annotations/Oriented Bounding Boxes/{stem}.xml"))
                except KeyError:
                    continue
                label = out_root / "labels" / split / f"{stem}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("".join(f"{c} " + " ".join(f"{v:.8f}" for pt in pts for v in pt) + "\n"
                                         for c, pts in rows), encoding="utf-8")
                if not a.skip_images:
                    dst = out_root / "images" / split / f"{stem}.jpg"
                    member = f"JPEGImages-trainval/{stem}.jpg"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not (dst.exists() and dst.stat().st_size == imgs.getinfo(member).file_size):
                        with imgs.open(member) as fin, dst.open("wb") as fout:
                            shutil.copyfileobj(fin, fout, length=1 << 20)
                images += 1
                objects += len(rows)
            stats[split] = {"images": images, "objects": objects}
    (out_root / "dior_r_obb.yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
