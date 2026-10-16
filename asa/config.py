"""The experiment description (``configs/experiments.json``): datasets, detectors and view layout.

Paths are relative to the working directory (run the tools from the repository root) unless
absolute. Exported views of detector ``D`` on split ``S`` live in

    {views_root}/D/S/base/<view>     views at the base input size (r0 ... h270)
    {views_root}/D/S/<scale>/<view>  views at another scale, e.g. 0.83/h0 for the view h0@0.83

A detector may instead list its view directories explicitly under ``"view_dirs"``
(``{split: {view: path}}``).
"""
import json
from pathlib import Path

from .views import FAMILIES


class Experiments:
    def __init__(self, path="configs/experiments.json"):
        self.path = Path(path)
        self.cfg = json.loads(self.path.read_text(encoding="utf-8"))
        self.data_root = Path(self.cfg.get("data_root", "data"))
        self.views_root = Path(self.cfg.get("views_root", "views"))
        self.datasets = self.cfg["datasets"]
        self.detectors = self.cfg["detectors"]

    def dataset_of(self, det):
        return self.detectors[det]["dataset"]

    def ds(self, det):
        return self.datasets[self.dataset_of(det)]

    def labels(self, det, split):
        return self.data_root / self.ds(det)["root"] / "labels" / split

    def images(self, det, split):
        return self.data_root / self.ds(det)["root"] / "images" / split

    def num_classes(self, det):
        return int(self.ds(det)["num_classes"])

    def view_dir(self, det, split, view):
        explicit = self.detectors[det].get("view_dirs", {}).get(split, {})
        if view in explicit:
            return Path(explicit[view])
        base, _, scale = view.partition("@")
        return self.views_root / det / split / (scale or "base") / base

    def view_dirs(self, det, split, family):
        return [self.view_dir(det, split, v) for v in FAMILIES[family]]

    def detectors_of(self, dataset):
        return [d for d, v in self.detectors.items() if v["dataset"] == dataset]

    @property
    def development(self):
        return list(self.cfg.get("development", []))
