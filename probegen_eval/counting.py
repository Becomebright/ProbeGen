"""Counting: count red dots that the generator added to the input image."""

from io import BytesIO
import re

import cv2
import numpy as np
from PIL import Image

from .generate import prepare_input


def score(dataset, sample, data_root, path):
    if path.suffix == ".txt":   # MLLM answer: first integer in the text, 0 if none
        match = re.search(r"-?\d+", path.read_text())
        return {"prediction": int(match.group()) if match else 0, "ground_truth": sample.target["count"]}
    with Image.open(BytesIO(prepare_input(data_root / sample.image))) as decoded:
        inp = decoded.convert("RGB")   # the exact input the generator received
    with Image.open(path) as image:
        gen = np.asarray(image.convert("RGB").resize(inp.size, Image.Resampling.BILINEAR))
    hsv = cv2.cvtColor(gen, cv2.COLOR_RGB2HSV)
    red = (cv2.inRange(hsv, (0, 160, 100), (10, 255, 255)) |
           cv2.inRange(hsv, (170, 160, 100), (180, 255, 255))) > 0
    diff = np.abs(gen.astype(np.int16) - np.asarray(inp).astype(np.int16)).sum(axis=-1)
    mask = ((diff > 60) & red).astype(np.uint8)
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold, params.maxThreshold = 10, 255
    params.filterByColor, params.blobColor = True, 255
    params.filterByArea, params.minArea, params.maxArea = True, 12, 1500
    params.filterByCircularity, params.minCircularity = True, 0.6
    params.filterByInertia, params.minInertiaRatio = True, 0.5
    params.filterByConvexity = False
    count = len(cv2.SimpleBlobDetector_create(params).detect(mask * 255))
    return {"prediction": count, "ground_truth": sample.target["count"]}


def aggregate(rows):
    errors = np.array([r["prediction"] - r["ground_truth"] for r in rows], dtype=float)
    return {"MAE": float(np.abs(errors).mean()), "RMSE": float(np.sqrt((errors ** 2).mean()))}
