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
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, Any
from dataclasses import dataclass
import json as _json

import numpy as _np
import numpy.typing as _npt
import scipy.io as _sio
import pandas as _pd
import h5py as _h5
import imageio.v3 as _iio
import cv2 as _cv2

from .typing import (
    PathLike,
    Hemisphere,
    ROIFileType,
)
from . import (
    landmarks as _landmarks,
)


DATA_DIR_NAME = "data"
ATLAS_MASK_FILE = "reference_masks.h5"
LEFT_OUTLINE_FILE = "atlas_outline_left.png"
RIGHT_OUTLINE_FILE = "atlas_outline_right.png"
REFERENCE_NAME = '__reference__'


@dataclass
class ROI:
    """manages a single ROI of an image."""
    name: str
    side: Hemisphere
    AllenID: int
    description: str
    mask: _npt.NDArray

    def _write_hdf(
        self,
        parent: _h5.Group,
        **options
    ):
        if self.side in ('left', 'right'):
            side = self.side
            if side not in parent.keys():
                parent.create_group(side)
            hemi = parent[side]
        else:
            hemi = parent

        if self.name in hemi.keys():
            del hemi[self.name]
        entry = hemi.create_dataset(
            self.name,
            data=self.mask.astype(_np.uint8),
            **options
        )
        entry.attrs['name'] = self.name
        entry.attrs['side'] = self.side
        if self.AllenID >= 0:
            entry.attrs['AllenID'] = self.AllenID
        entry.attrs['description'] = self.description
    
    def _write_metadata(self, metadata: Dict[str, Any]):
        if self.name in metadata.keys():
            return
        metadata[self.name] = dict()
        metadata[self.name]['name'] = self.name
        if self.AllenID >= 0:
            metadata[self.name]['AllenID'] = self.AllenID
        metadata[self.name]['description'] = self.description
    
    def _write_data(self, data: Dict[str, Any]):
        if self.side in ('left', 'right'):
            if self.side not in data.keys():
                data[self.side] = dict()
            hemi = data[self.side]
        else:
            hemi = data
        hemi[self.name] = self.mask.astype(_np.uint8)


@dataclass
class ROISet:
    """manages a set of ROIs of an image."""
    image_name: str
    frame_idx: int
    total_frames: int
    rois: Tuple[ROI]

    @property
    def names(self) -> Tuple[str]:
        """the names of the ROIs, as it has been registered to this ROISet.
        Duplicates are supposed to be removed."""
        names = []
        for roi in self.rois:
            if roi.name not in names:
                names.append(roi.name)
        return tuple(names)

    def to_hdf(self, path: PathLike, **options):
        """`options` can be used to pass to `create_dataset`"""
        with _h5.File(str(path), 'w') as out:
            out.attrs['image_name'] = self.image_name
            out.attrs['frame_idx']  = self.frame_idx

            rois = out.create_group('rois')
            rois.attrs['names']  = _json.dumps(self.names, indent=None)
            for roi in self.rois:
                roi._write_hdf(out, **options)
    
    def to_matfile(self, path: PathLike):
        metadata = dict()
        metadata['image_name'] = self.image_name
        metadata['frame_idx']  = self.frame_idx
        metadata['rois'] = dict()
        data = dict()
        for roi in self.rois:
            roi._write_metadata(metadata['rois'])
            roi._write_data(data)
        _sio.savemat(str(path), {
            'metadata': metadata,
            'rois': data
        })
    
    def to_file(self, path: PathLike, filetype: ROIFileType):
        if filetype == 'hdf':
            self.to_hdf(path)
        elif filetype == 'matlab':
            self.to_matfile(path)
        else:
            raise ValueError(f"ROI file type expected to be one of ('hdf', 'matlab'), but got {repr(filetype)}")


def generate_rois_batch(
    alignment: Tuple[_landmarks.Alignment],
    metadata: _pd.DataFrame,
    reference: Optional[ROISet] = None,
    outline: Optional[ROISet] = None,
    resize: bool = True
) -> Tuple[ROISet]:
    if reference is None:
        reference = load_reference_ROIs()
    if outline is None:
        outline = load_reference_outlines()
    sets = []
    for rowidx, row in metadata.iterrows():
        sets.append(
            generate_rois_single(
                alignment[rowidx],
                row.to_dict(),
                reference=reference,
                outline=outline,
                resize=resize
            )
        )
    return tuple(sets)


def generate_rois_single(
    alignment: _landmarks.Alignment,
    metadata: Dict[str, Union[int, str]],
    reference: Optional[ROISet] = None,
    outline: Optional[ROISet] = None,
    resize: bool = True
) -> ROISet:

    def _warp_ROI(
        roi: ROI,
        alignment: _landmarks.Alignment,
        shape: Optional[Tuple[int, int]] = None
    ) -> ROI:
        mask = alignment.warp_image(roi.mask, side=roi.side)
        if shape is not None:
            # NOTE: `512` being the 'standard' size used in the pipeline
            method = _cv2.INTER_CUBIC if max(shape) >= 512 else _cv2.INTER_AREA
            mask = _cv2.resize(mask, shape, interpolation=method)
        return ROI(
            name=roi.name,
            side=roi.side,
            AllenID=roi.AllenID,
            description=roi.description,
            mask=mask
        )

    if reference is None:
        reference = load_reference_ROIs()
    if outline is None:
        outline = load_reference_outlines()
    shape = (metadata['Width'], metadata['Height']) if bool(resize) else None
    
    warped_outlines = tuple(
        _warp_ROI(roi, alignment, shape=shape) for roi in outline.rois
    )
    warped_rois = tuple(
        _warp_ROI(roi, alignment, shape=shape) for roi in reference.rois
    )
    merged_outline = ROI(
        name='outline',
        side='both',
        AllenID=-1,
        description='the expected outline of the brain for this ROI set',
        mask=_np.logical_or(warped_outlines[0].mask, warped_outlines[1].mask)
    )
    return ROISet(
        image_name=metadata['Image'],
        frame_idx=metadata['Frame'],
        total_frames=metadata['TotalFrames'],
        rois=(merged_outline,) + tuple(warped_outlines) + tuple(warped_rois)
    )


def load_reference_ROIs() -> ROISet:
    h5path = default_reference_ROI_path()
    rois = []
    with _h5.File(str(h5path), 'r') as src:
        masks = src['masks']
        for idx in sorted(masks.keys()):
            entry = masks[idx]
            mask = _np.array(entry)
            metadata = dict((k, v) for k, v in entry.attrs.items())
            rois.append(ROI(mask=mask, **metadata))
    return ROISet(image_name=REFERENCE_NAME, frame_idx=-1, total_frames=-1, rois=tuple(rois))


def load_reference_outlines() -> ROISet:
    left_outline_path, right_outline_path = default_outline_paths()
    left_outline = _iio.imread(str(left_outline_path))
    right_outline = _iio.imread(str(right_outline_path))
    if left_outline.ndim == 3:
        left_outline = left_outline.mean(-1)
        right_outline = right_outline.mean(-1)
    outlines = {
        'left': (left_outline > 0),
        'right': (right_outline > 0),
    }
    rois = []
    for side, outline in outlines.items():
        rois.append(ROI(
            name='outline',
            side=side,
            AllenID=-1,
            description='the expected outline of the brain for this ROI set',
            mask=outline.astype(_np.uint8) * 255
        ))
    return ROISet(image_name=REFERENCE_NAME, frame_idx=-1, total_frames=-1, rois=tuple(rois))


def default_reference_ROI_path() -> Path:
    datadir = Path(__file__).parent / DATA_DIR_NAME
    return datadir / ATLAS_MASK_FILE


def default_outline_paths() -> Tuple[Path, Path]:
    datadir = Path(__file__).parent / DATA_DIR_NAME
    left_outline = datadir / LEFT_OUTLINE_FILE
    right_outline = datadir / RIGHT_OUTLINE_FILE
    return left_outline, right_outline
