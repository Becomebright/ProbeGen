"""Segmentation: Otsu binarization and IoU."""

import json

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from skimage.filters import threshold_otsu
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


def load_gray(path, shape):
    """Grayscale prediction resized to the ground-truth shape."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("L")).copy()
    tensor = torch.from_numpy(array)[None, None]
    return resize(tensor, list(shape), interpolation=InterpolationMode.BILINEAR,
                  antialias=True)[0, 0].numpy()


def score(dataset, sample, data_root, path):
    target = load_target(dataset, sample, data_root)
    prediction = load_gray(path, target.shape)
    prediction = prediction > threshold_otsu(prediction)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    iou = float(np.float32(intersection) / (np.float32(union) + np.float32(1e-5))) if union else 1.0
    return {"iou": iou, "intersection": intersection, "union": union}


def aggregate(rows):
    return {"gIoU": float(np.mean([r["iou"] for r in rows])),
            "cIoU": sum(r["intersection"] for r in rows) / (sum(r["union"] for r in rows) + 1e-10)}


def load_target(dataset, sample, data_root):
    target = sample.target
    if dataset == "ReasonSeg":
        # cv2.imread applies EXIF orientation, which the annotations assume.
        height, width = cv2.imread(str(data_root / sample.image)).shape[:2]
        return reasonseg_mask(data_root / target["annotation"], (height, width))
    if dataset == "DRAM":
        with Image.open(data_root / target["label"]) as image:
            return np.asarray(image) == target["class_id"]
    # RefCOCO: polygons; Manga109: RLE.
    seg, h, w = target["segmentation"], target["height"], target["width"]
    decoded = coco_mask.decode(coco_mask.frPyObjects(seg, h, w) if isinstance(seg, list) else seg)
    return decoded.any(axis=2) if decoded.ndim == 3 else decoded > 0


def reasonseg_mask(path, shape):
    """Rasterize polygons from largest to smallest; "ignore" regions count as background."""
    shapes = [s for s in json.loads(path.read_text())["shapes"] if s["label"].lower() != "flag"]
    polygons = [np.array([s["points"]], dtype=np.int32) for s in shapes]

    def draw(mask, polygon, value):
        cv2.polylines(mask, polygon, True, value, 1)
        cv2.fillPoly(mask, polygon, value)
        return mask

    areas = [draw(np.zeros(shape, np.uint8), p, 1).sum() for p in polygons]
    mask = np.zeros(shape, np.uint8)
    for i in np.argsort(areas)[::-1]:
        draw(mask, polygons[i], 255 if "ignore" in shapes[i]["label"].lower() else 1)
    return mask == 1
