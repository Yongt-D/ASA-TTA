"""Oriented boxes as four-vertex polygons: file I/O, polygon IoU and image sizes.

Prediction and label files hold one box per line,

    cls x1 y1 x2 y2 x3 y3 x4 y4 [conf]

with the vertices normalised by the image width and height (the YOLO-OBB label format).
A box is passed around as a dict ``{"cls", "poly", "bounds", "conf"}``.
"""
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def polygon_orientation(poly):
    signed = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        signed += x1 * y2 - x2 * y1
    return 1.0 if signed >= 0 else -1.0


def _inside(point, edge_start, edge_end, orient):
    x, y = point
    x1, y1 = edge_start
    x2, y2 = edge_end
    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return orient * cross >= -1e-12


def _line_intersection(p1, p2, q1, q2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return px, py


def polygon_clip(subject, clip):
    """Sutherland-Hodgman clipping of ``subject`` by the convex polygon ``clip``."""
    output = list(subject)
    orient = polygon_orientation(clip)
    for i, edge_start in enumerate(clip):
        edge_end = clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        prev = input_list[-1]
        for curr in input_list:
            curr_inside = _inside(curr, edge_start, edge_end, orient)
            prev_inside = _inside(prev, edge_start, edge_end, orient)
            if curr_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, curr, edge_start, edge_end))
                output.append(curr)
            elif prev_inside:
                output.append(_line_intersection(prev, curr, edge_start, edge_end))
            prev = curr
    return output


def polygon_iou(poly_a, poly_b):
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter = polygon_area(polygon_clip(poly_a, poly_b))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def poly_bounds(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def aabb_iou(bounds_a, bounds_b):
    ax1, ay1, ax2, ay2 = bounds_a
    bx1, by1, bx2, by2 = bounds_b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def fast_iou(item_a, item_b, aabb_prefilter=0.05):
    """Polygon IoU of two boxes; pairs whose axis-aligned IoU is below the prefilter count as 0."""
    if aabb_iou(item_a["bounds"], item_b["bounds"]) < aabb_prefilter:
        return 0.0
    return polygon_iou(item_a["poly"], item_b["poly"])


def read_obb_file(path, is_pred):
    """Boxes of one label (``is_pred=False``, conf 1) or prediction file; missing file -> []."""
    path = Path(path)
    items = []
    if not path.exists():
        return items
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        nums = [float(x) for x in parts[1:]]
        if len(nums) < 8:
            continue
        conf = nums[8] if is_pred and len(nums) >= 9 else 1.0
        poly = [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]
        items.append({"cls": cls, "poly": poly, "bounds": poly_bounds(poly), "conf": conf})
    return items


def write_obb_file(path, preds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for pred in preds:
        coords = [c for xy in pred["poly"] for c in xy]
        row = [str(int(pred["cls"]))] + [f"{v:.8g}" for v in coords] + [f"{float(pred['conf']):.8g}"]
        lines.append(" ".join(row))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def image_aspects(img_dir, stems=None):
    """Image stem -> height / width, read from the image headers (only ``stems`` if given)."""
    from PIL import Image
    out = {}
    wanted = None if stems is None else set(stems)
    for p in Path(img_dir).iterdir():
        if p.suffix.lower() in IMAGE_SUFFIXES and (wanted is None or p.stem in wanted):
            with Image.open(p) as im:
                w, h = im.size
            out[p.stem] = h / w
    return out
