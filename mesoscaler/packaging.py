# MIT License
#
# Copyright (c) 2023-2024 Keisuke Sehara
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

from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import numpy.typing as _npt
import scipy.io as _sio

from . import (
    landmarks as _landmarks,
    rois as _rois,
)
from .typing import (
    PathLike,
)
from .libwrapper import (
    h5py as _h5
)


ResultsDataType = Literal['512', 'original']


@dataclass
class ResultsPackageIndices:
    images: str = 'images'
    landmarks: str = 'landmarks'
    alignment: str = 'affine_ref_to_data'
    rois: str = 'rois'


@dataclass
class ResultImages:
    source: _npt.NDArray
    landmarks: _npt.NDArray
    alignment: _npt.NDArray


@dataclass
class Results:
    name: str
    images: ResultImages
    landmarks: _landmarks.Landmarks
    alignment: _landmarks.Alignment
    rois: _rois.ROISet
    datatype: ResultsDataType = '512'

    def package_keys(self) -> ResultsPackageIndices:
        indices = ResultsPackageIndices()
        if self.datatype == '512':
            indices.images += '512'
            indices.landmarks += '512'
            indices.alignment += '512'
        return indices


def package_matfile(
    results: Results,
    output_dir: PathLike
):
    # FIXME:
    # is the difference in array orderings taken into account?
    #
    output_dir = Path(output_dir)
    keys = results.package_keys()
    data = dict()
    data['metadata'] = results.rois.metadata_dict(with_roi_metadata=True)
    data[keys.images] = {
        'source': results.images.source,
        'landmarks': results.images.landmarks,
        'alignment': results.images.alignment
    }
    data[keys.landmarks] = results.landmarks.to_dict()
    if results.alignment.separate:
        data[keys.alignment] = {
            'left': results.alignment.left,
            'right': results.alignment.right
        }
    else:
        data[keys.alignment] = results.alignment.left
    data[keys.rois] = results.rois.data_dict()

    name    = Path(results.name).stem
    outpath = (output_dir / f"{name}_mesoscaler.mat")
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)
    _sio.savemat(
        str(outpath),
        data
    )


def package_hdf(
    results: Results,
    output_dir: PathLike,
    **options
):
    output_dir = Path(output_dir)
    name    = Path(results.name).stem
    outpath = (output_dir / f"{name}_mesoscaler.h5")
    keys = results.package_keys()
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)
    with _h5.File(str(outpath), 'w') as out:
        for k, v in results.rois.metadata_dict(with_roi_metadata=False).items():
            out.attrs[k] = v
        images = out.create_group(keys.images)
        for k in ('source', 'landmarks', 'alignment'):
            images.create_dataset(k, data=getattr(results.images, k), **options)
        results.landmarks.to_hdf(
            out,
            key=keys.landmarks,
            **options
        )
        results.alignment.to_hdf(
            out,
            key=keys.alignment,
            **options
        )
        results.rois.to_hdf(
            out,
            group_key=keys.rois,
            write_metadata=False,
            **options
        )
