# MIT License
#
# Copyright (c) 2023 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Iterable, Tuple
from pathlib import Path
import math as _math

import numpy as _np
import numpy.typing as _npt
import cv2 as _cv2

from . import (
    defaults as _defaults,
)
from .typing import (
    ROIFileType,
)


def unique_names_from_path(paths: Iterable[Path]) -> Tuple[str]:
    NUM_MAX_TEST = 10

    def with_level(i):
        return [str(path.relative_to(path.parents[i])) for path in paths]

    def is_unique(names):
        return len(set(names)) == len(names)

    for i in range(NUM_MAX_TEST):
        names = with_level(i)
        if is_unique(names):
            return tuple(names)
    raise RuntimeError('failed to find unique names from a set of paths')


def required_number_of_digits(total: int, minimum: int = 2) -> int:
    required = _math.floor(_math.log10(total)) + 1
    return max(required, minimum)


def get_roi_file_suffix(filetype: ROIFileType) -> str:
    if filetype == 'hdf':
        return '.h5'
    elif filetype == 'matlab':
        return '.mat'
    else:
        raise ValueError(f"ROI file type expected to be one of ('hdf', 'matlab'), got {repr(filetype)}")


def estimate_affine(
    src: _npt.NDArray,
    dst: _npt.NDArray
) -> _npt.NDArray:
    """given set of points `src` and `dst`, estimate the
    Affine transformation matrix `A` that converts `src` to `dst`.

    Both `src` and `dst` are the N x 2-D points, in shape (N, 2).
    `src[i]` and `dst[i]` must match each other."""
    N   = src.shape[0]
    assert dst.shape[0] == N
    src = _np.concatenate([src, _np.ones((N, 1), dtype=src.dtype)], axis=1)  # (N, 3)
    X_in = _np.concatenate([src, _np.zeros((N, 6), dtype=src.dtype), src], axis=1)  # (N, 12)
    X_in = X_in.reshape((N, 2, 6), order='C')  # (N, 2, 6)
    X_in = X_in.reshape((N * 2, 6), order='C')  # (N*2, 6)
    x_out = dst.reshape((-1, 1), order='C')  # (N*2, 1)
    a, _, _, _ = _np.linalg.lstsq(X_in, x_out, rcond=None)  # (6, 1): the rest are `residuals`, `rank` and `s`
    return a.reshape((2, 3), order='C')


def affine_warp_image(
    img: _npt.NDArray,
    warp: _npt.NDArray,
    size: Optional[Tuple[int]] = None
) -> _npt.NDArray:
    """applies the warp matrix `warp` to the given image `img`."""
    if size is None:
        size = _defaults.VIDEO_FRAME_SIZE
    return _cv2.warpAffine(img, warp, dsize=size)


def affine_warp_points(
    pts: _npt.NDArray,
    warp: _npt.NDArray
) -> _npt.NDArray:
    """applies the warp matrix `warp` (shape (2, 3)) to the given set of 2-d points,
    `pts`, in shape (num_points, 2)."""
    [N, K] = pts.shape
    assert K == 2
    pts = _np.concatenate([pts, _np.ones((N, 1))], axis=1)
    return (pts @ warp.T)
