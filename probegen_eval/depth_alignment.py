# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from Marigold (src/util/alignment.py) for ProbeGen: unused options removed.
import numpy as np


def align_depth_least_square(gt, pred, valid_mask):
    """Fit scale and shift so that pred * scale + shift matches gt on valid pixels.

    Fit in depth space ("least_square") or pass disparities ("least_square_disparity").
    Returns the aligned prediction, scale and shift.
    """
    x = pred[valid_mask].reshape((-1, 1))
    A = np.concatenate([x, np.ones_like(x)], axis=-1)
    scale, shift = np.linalg.lstsq(A, gt[valid_mask].reshape((-1, 1)), rcond=None)[0]
    return pred * scale + shift, scale, shift


def depth2disparity(depth, return_mask=False):
    disparity = np.zeros_like(depth)
    positive = depth > 0
    disparity[positive] = 1.0 / depth[positive]
    return (disparity, positive) if return_mask else disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)
