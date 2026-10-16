"""TTA views and the exact mapping of predictions back to the original image.

A view is named ``r{0,90,180,270}`` (a counter-clockwise rotation by ``np.rot90``) or
``h{0,90,180,270}`` (the same rotation followed by a left-right flip); ``h0`` is the left-right
flip and ``h180`` the up-down flip. A view at another input size carries a suffix, e.g.
``h0@0.83`` for the flipped pass of the multi-scale family at 0.83 times the base size.

Box coordinates are continuous: pixel ``i`` spans ``[i, i + 1)``, so an image of width ``W``
spans ``[0, W]`` and the inverse of a quarter turn is ``x -> W - y`` without ``- 1``. The
pixel-index inverse (``W - 1 - y``) is exact for integer pixel positions only and shifts
continuous box coordinates by one pixel; exporters therefore record the mapping they used
(``back_mapping`` in ``export.json``) and the cluster builder refuses views without it.
Run ``python -m asa.views`` for a self-test against ``np.rot90``.
"""
import json
from pathlib import Path

import numpy as np

BACK_MAPPING = "continuous"

FAMILIES = {
    "flip": ["r0", "h0"],
    "d2": ["r0", "h0", "h180", "r180"],
    "c4": ["r0", "r90", "r180", "r270"],
    "d4": ["r0", "r90", "r180", "r270", "h0", "h90", "h180", "h270"],
    "ms": ["r0", "h0@0.83", "r0@0.67"],  # the multi-scale scheme of YOLOv5
}


def rotate_points(points, k, width, height):
    """Where continuous points of the original image land in ``np.rot90(image, k)``."""
    out = np.array(points, dtype=float, copy=True)
    x = out[..., 0].copy()
    y = out[..., 1].copy()
    if k == 0:
        return out
    if k == 1:
        out[..., 0], out[..., 1] = y, width - x
    elif k == 2:
        out[..., 0], out[..., 1] = width - x, height - y
    elif k == 3:
        out[..., 0], out[..., 1] = height - y, x
    else:
        raise ValueError(f"k must be in [0, 3], got {k}")
    return out


def inverse_rotate_points(points, k, width, height):
    """Map continuous coordinates from ``np.rot90(image, k)`` back to the original image."""
    out = np.array(points, dtype=float, copy=True)
    x = out[..., 0].copy()
    y = out[..., 1].copy()
    if k == 0:
        return out
    if k == 1:
        out[..., 0], out[..., 1] = width - y, x
    elif k == 2:
        out[..., 0], out[..., 1] = width - x, height - y
    elif k == 3:
        out[..., 0], out[..., 1] = y, height - x
    else:
        raise ValueError(f"k must be in [0, 3], got {k}")
    return out


def view_image(image, k, flip):
    """The view ``r{90k}`` (flip=False) or ``h{90k}`` (flip=True): np.rot90, then a left-right flip."""
    out = np.rot90(image, k)
    return np.fliplr(out) if flip else out


def inverse_view_points(points, k, flip, width, height):
    """Map continuous coordinates from ``view_image(image, k, flip)`` back to the original image."""
    out = np.array(points, dtype=float, copy=True)
    if flip:
        view_width = width if k % 2 == 0 else height
        out[..., 0] = view_width - out[..., 0]
    return inverse_rotate_points(out, k, width, height)


def parse_view(name):
    """``'r90'`` -> (1, False), ``'h270@0.83'`` -> (3, True); the scale suffix is ignored here."""
    base = name.split("@")[0]
    if len(base) < 2 or base[0] not in "rh" or not base[1:].isdigit() or int(base[1:]) % 90:
        raise ValueError(f"view must look like r0/r90/.../h270[@scale], got {name!r}")
    return (int(base[1:]) // 90) % 4, base[0] == "h"


def check_backmapping(view_dir):
    """Refuse a rotated or flipped view whose export does not record the continuous mapping."""
    view_dir = Path(view_dir)
    if view_dir.name == "r0":
        return
    for name in ("export.json", "c4_export.json"):
        manifest = view_dir.parent / name
        if manifest.exists():
            recorded = str(json.loads(manifest.read_text()).get("back_mapping", ""))
            if recorded.startswith(BACK_MAPPING):
                return
            raise SystemExit(f"{view_dir}: back_mapping={recorded!r}, expected {BACK_MAPPING!r}")
    raise SystemExit(f"{view_dir}: no export.json next to the view; export it with tools/export_views_*.py")


def selftest():
    rng = np.random.default_rng(2027)
    shapes = ((7, 11), (11, 7), (8, 8), (5, 9), (1024, 1024), (1024, 525), (863, 1024))
    for height, width in shapes:
        for k in range(4):
            image = np.zeros((height, width), dtype=np.uint8)
            x0 = int(rng.integers(0, width - 1))
            y0 = int(rng.integers(0, height - 1))
            x1 = int(rng.integers(x0 + 1, width + 1))
            y1 = int(rng.integers(y0 + 1, height + 1))
            image[y0:y1, x0:x1] = 1
            for flip in (False, True):
                rows, cols = np.nonzero(view_image(image, k, flip))
                vx0, vy0, vx1, vy1 = int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1
                corners = np.array([[vx0, vy0], [vx1, vy0], [vx1, vy1], [vx0, vy1]], dtype=float)
                back = inverse_view_points(corners, k, flip, width, height)
                got = (back[:, 0].min(), back[:, 1].min(), back[:, 0].max(), back[:, 1].max())
                if got != (x0, y0, x1, y1):
                    raise AssertionError(f"{'h' if flip else 'r'}{90 * k} on {height}x{width}: {got}")
            pts = rng.uniform(0, 1, (32, 4, 2)) * [width, height]
            if np.abs(inverse_rotate_points(rotate_points(pts, k, width, height), k, width, height) - pts).max() > 1e-9:
                raise AssertionError(f"round trip failed, k={k} on {height}x{width}")
    print(f"views selftest OK: {len(shapes)} image shapes x 8 views")


if __name__ == "__main__":
    selftest()
