"""Prepare DOTA-v1.0 for the YOLO-OBB pipeline: labels, then 1024-pixel crops with a 200-pixel gap.

    python tools/prepare_dota.py --src-root data/dota_v1/DOTA_V1.0 --out-root data/dota_yolo_obb_1024

``--src-root`` holds ``{train,val}/images/*.png`` and ``{train,val}/labelTxt/*.txt`` (the
DOTA-v1.0 annotation files). Labels are normalised by the image size and written next to a
link to the images under ``--raw-root``; Ultralytics' ``split_images_and_labels`` then cuts
the crops. Crop labels are clipped to [0, 1] and degenerate boxes dropped. Writes
``<out-root>.yaml``. The val split has 5,297 crops, of which the 3,066 with a label file are
evaluated.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2

DOTA_CLASSES = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court",
                "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter",
                "roundabout", "soccer-ball-field", "swimming-pool"]
CLASS_TO_IDX = {name: i for i, name in enumerate(DOTA_CLASSES)}


def link_dir(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists():
        return
    try:
        link.symlink_to(target.resolve(), target_is_directory=True)
    except OSError:  # Windows without the symlink privilege: a directory junction
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target.resolve())], check=True,
                       stdout=subprocess.DEVNULL)


def convert_labels(src_root: Path, raw_root: Path, split: str) -> None:
    image_dir, label_dir = src_root / split / "images", src_root / split / "labelTxt"
    out_dir = raw_root / "labels" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    link_dir(raw_root / "images" / split, image_dir)
    n_files = n_obj = 0
    for image_path in sorted(image_dir.glob("*.png")):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        lines = []
        if label_path.exists():
            for line in label_path.read_text(errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) < 9 or parts[8] not in CLASS_TO_IDX:
                    continue  # also skips the imagesource / gsd header lines
                try:
                    coords = [float(v) for v in parts[:8]]
                except ValueError:
                    continue
                norm = [min(max(v / (w if i % 2 == 0 else h), 0.0), 1.0) for i, v in enumerate(coords)]
                lines.append(f"{CLASS_TO_IDX[parts[8]]} " + " ".join(f"{v:.6g}" for v in norm))
        (out_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_files += 1
        n_obj += len(lines)
    print(f"{split}: {n_files} label files, {n_obj} objects", flush=True)


def polygon_area(coords: list[float]) -> float:
    xs, ys = coords[0::2], coords[1::2]
    return abs(sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i] for i in range(4))) * 0.5


def sanitize_labels(label_dir: Path) -> None:
    n_files = n_clipped = n_dropped = 0
    for label_path in label_dir.glob("*.txt"):
        lines, changed = [], False
        for raw in label_path.read_text(errors="ignore").splitlines():
            parts = raw.strip().split()
            try:
                cls, coords = int(float(parts[0])), [float(v) for v in parts[1:]]
            except (ValueError, IndexError):
                cls, coords = -1, []
            if len(coords) != 8:
                changed, n_dropped = True, n_dropped + 1
                continue
            clipped = [min(max(v, 0.0), 1.0) for v in coords]
            if clipped != coords:
                changed, n_clipped = True, n_clipped + 1
            if not 0 <= cls < len(DOTA_CLASSES) or polygon_area(clipped) <= 1e-8:
                changed, n_dropped = True, n_dropped + 1
                continue
            lines.append(f"{cls} " + " ".join(f"{v:.6g}" for v in clipped))
        if changed:
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_files += 1
    print(f"{label_dir}: {n_files} files changed, {n_clipped} boxes clipped, {n_dropped} dropped", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="data/dota_v1/DOTA_V1.0")
    ap.add_argument("--raw-root", default="data/dota_yolo_obb_raw")
    ap.add_argument("--out-root", default="data/dota_yolo_obb_1024")
    ap.add_argument("--crop-size", type=int, default=1024)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    from ultralytics.data.split_dota import split_images_and_labels

    src_root, raw_root, out_root = Path(a.src_root), Path(a.raw_root), Path(a.out_root)
    if a.rebuild and out_root.exists():
        shutil.rmtree(out_root)
    for split in a.splits:
        convert_labels(src_root, raw_root, split)
        out_images = out_root / "images" / split
        if a.rebuild or not out_images.exists() or not any(out_images.iterdir()):
            print(f"{split}: cutting {a.crop_size}-pixel crops, gap {a.gap}", flush=True)
            split_images_and_labels(str(raw_root), str(out_root), split=split, crop_sizes=(a.crop_size,), gaps=(a.gap,))
        sanitize_labels(out_root / "labels" / split)
        print(f"{split}: {sum(1 for _ in out_images.iterdir())} crops, "
              f"{sum(1 for _ in (out_root / 'labels' / split).glob('*.txt'))} label files", flush=True)
    names = "".join(f"  {i}: {n.replace('-', ' ')}\n" for i, n in enumerate(DOTA_CLASSES))
    out_root.with_suffix(".yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}", encoding="utf-8")


if __name__ == "__main__":
    main()
