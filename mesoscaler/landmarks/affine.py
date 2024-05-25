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
"""functions related to Affine transform."""
from typing import Optional, Tuple, Union

import numpy as _np
import numpy.typing as _npt
import cv2 as _cv2

from .. import (
    defaults as _defaults,
)


AffineCompact = _npt.NDArray
AffineSquare  = _npt.NDArray
AffineMatrix  = Union[AffineCompact, AffineSquare]


def identity() -> AffineCompact:
    return to_compact(_np.eye(3))


def estimate(
    src: _npt.NDArray,
    dst: _npt.NDArray
) -> AffineCompact:
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


def to_square(compact: AffineMatrix) -> AffineSquare:
    nrows, ncols = compact.shape
    if nrows == 3:
        return compact
    else:
        return _np.vstack([compact, [0, 0, 1]], dtype=_np.float32)


def to_compact(square: AffineMatrix) -> AffineCompact:
    nrows, ncols = square.shape
    if nrows == 3:
        return square[:2, :]
    else:
        return square


def invert(M: AffineMatrix) -> AffineCompact:
    return _cv2.invertAffineTransform(to_compact(M))


def compose(*matrices: Tuple[AffineMatrix]) -> AffineCompact:
    """composes the affine transformations.
    ``compose(A, B, C)`` results in the transformation in the
    order ``A -> B -> C``"""
    M = _np.eye(3)
    if len(matrices) == 0:
        return M
    elif len(matrices) == 1:
        return to_compact(matrices[0])
    else:
        for A in matrices:
            M = to_square(A) @ M
        return to_compact(M)


def warp_image(
    img: _npt.NDArray,
    warp: AffineMatrix,
    size: Optional[Tuple[int]] = None
) -> _npt.NDArray:
    """applies the warp matrix `warp` to the given image `img`."""
    if size is None:
        size = _defaults.VIDEO_FRAME_SIZE
    return _cv2.warpAffine(img, to_compact(warp), dsize=size)


def warp_points(
    pts: _npt.NDArray,
    warp: AffineMatrix,
) -> _npt.NDArray:
    """applies the warp matrix `warp` (shape (2, 3)) to the given set of 2-d points,
    `pts`, in shape (num_points, 2)."""
    [N, K] = pts.shape
    assert K == 2
    pts = _np.concatenate([pts, _np.ones((N, 1))], axis=1)
    return (pts @ to_square(warp).T)[:, :2]
