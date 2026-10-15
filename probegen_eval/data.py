"""Benchmark manifests."""

from dataclasses import dataclass, field
import json
from pathlib import Path

TASKS = {
    "nyuv2": "depth", "diode": "depth", "scannet": "depth", "kitti": "depth",
    "refcoco": "segmentation", "refcoco+": "segmentation", "refcocog": "segmentation",
    "ReasonSeg": "segmentation", "DRAM": "segmentation", "Manga109": "segmentation",
    "pixmo-points": "counting",
}


@dataclass
class Sample:
    id: str
    image: str          # input image, relative to the data root
    prediction: str     # prediction filename inside the output folder
    target: dict = field(default_factory=dict)
    query: str = ""


def load_manifest(path, n_samples=None):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = [Sample(**record) for record in doc["samples"]]
    return samples[:n_samples] if n_samples else samples
