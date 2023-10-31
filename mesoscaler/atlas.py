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

from typing import Optional, Tuple
import dataclasses as _dataclasses

import numpy.typing as _npt

from . import (
    landmarks as _landmarks,
)
from .typing import (
    PathLike,
)


@_dataclasses.dataclass
class Atlas:
    names: Tuple[str]
    masks: Tuple[_npt.NDArray]

    def to_hdf(self, outpath: PathLike):
        raise NotImplementedError()  # TODO
    
    def to_matfile(self, outpath: PathLike):
        raise NotImplementedError()  # TODO
    
    def to_png(self, outdir: PathLike):
        raise NotImplementedError()  # TODO


def load_reference_atlas(
    atlas_dir: Optional[PathLike] = None
) -> Atlas:
    raise NotImplementedError()  # TODO


def warp_atlas(
    atlas: Atlas,
    alignment: _landmarks.Alignment
) -> Atlas:
    raise NotImplementedError()  # TODO
