"""Depth: scale-and-shift alignment in disparity space, then AbsRel."""

import numpy as np
from PIL import Image
import torch

from .depth_alignment import align_depth_least_square, depth2disparity, disparity2depth
from .segmentation import load_gray


def score(dataset, sample, data_root, path):
    gt, valid, lower, upper = load_target(dataset, sample, data_root)
    pred = load_gray(path, gt.shape).astype(np.float32)   # brighter = closer
    # Least-squares scale and shift to ground-truth disparity (least_square_disparity).
    disparity, positive = depth2disparity(gt, return_mask=True)
    fit = valid & positive & (pred > 0)
    if not fit.any():
        raise ValueError("No positive prediction pixels for alignment")
    aligned, _, _ = align_depth_least_square(disparity, pred, fit)
    depth = np.clip(disparity2depth(np.clip(aligned, 1e-3, None)), lower, upper)
    p, g, v = torch.from_numpy(depth), torch.from_numpy(gt), torch.from_numpy(valid)
    error = torch.abs(p - g) / g
    error[~v] = 0
    return {"abs_rel": float(error.sum() / v.sum())}


def aggregate(rows):
    return {"AbsRel": float(np.mean([r["abs_rel"] for r in rows]))}


def load_target(dataset, sample, data_root):
    """Return ground-truth depth (m), validity mask and the clipping range."""
    if dataset == "diode":
        gt = np.load(data_root / sample.target["depth"]).squeeze().astype(np.float32)
        valid = np.load(data_root / sample.target["valid_mask"]).squeeze().astype(bool)
        return gt, valid, 0.6, 350.0
    with Image.open(data_root / sample.target["depth"]) as image:
        gt = np.asarray(image) / (256.0 if dataset == "kitti" else 1000.0)
    gt = gt.astype(np.float32)
    lower, upper = (1e-5, 80.0) if dataset == "kitti" else (1e-3, 10.0)
    if dataset == "kitti":
        h, w = gt.shape
        gt = gt[h - 352:, (w - 1216) // 2:(w - 1216) // 2 + 1216]   # KITTI benchmark crop
    valid = (gt > lower) & (gt < upper)
    if dataset in ("nyuv2", "kitti"):
        crop = np.zeros_like(valid)
        h, w = gt.shape
        if dataset == "nyuv2":
            crop[45:471, 41:601] = True                                # Eigen crop
        else:
            crop[int(.3324324 * h):int(.91351351 * h), int(.0359477 * w):int(.96405229 * w)] = True
        valid &= crop
    return gt, valid, lower, upper
