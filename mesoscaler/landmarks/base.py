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

# flake8: noqa: E712

from pathlib import Path
from typing import Optional, Union, Tuple, Iterable, Dict
import dataclasses as _dataclasses

import numpy as _np
import numpy.typing as _npt
import pandas as _pd
import cv2 as _cv2
import imageio.v3 as _iio
from PIL import (
    Image as _PILImage,
    ImageDraw as _PILDraw,
)

from ..typing import (
    PathLike,
    Number,
    Hemisphere,
)
from .. import (
    defaults as _defaults,
)
from . import (
    reference as _refs,
    paths as _paths,
    affine as _affine,
)

@_dataclasses.dataclass
class DLCOutput:
    table: _pd.DataFrame
    images: _npt.NDArray

    Self = 'DLCOutput'

    @property
    def size(self) -> int:
        return self.table.shape[0]

    @classmethod
    def from_directory(
        cls,
        input_dir: PathLike,
        load_images: bool = True
    ) -> Self:
        input_dir = Path(input_dir)
        tabpath = _paths.predicted_landmarks_table_path(input_dir)
        imgpath = _paths.predicted_landmarks_video_path(input_dir)
        if not tabpath.exists():
            raise FileNotFoundError('CSV file not found: ' + str(tabpath))
        elif (load_images == True) and (not imgpath.exists()):
            raise FileNotFoundError('video file not found: ' + str(imgpath))
        table = _pd.read_csv(tabpath, header=[0, 1, 2])
        if load_images == True:
            images = _iio.imread(str(imgpath))
        else:
            images = None
        return cls(table=table, images=images)


@_dataclasses.dataclass
class Landmark:
    name: str
    coords: _npt.NDArray

    @property
    def x(self) -> float:
        return self.coords[0]

    @property
    def y(self) -> float:
        return self.coords[1]

    @property
    def p(self) -> float:
        return self.coords[2]
    
    def is_valid(self) -> bool:
        return all(x for x in ~_np.isnan(self.coords.ravel()[:2]))


@_dataclasses.dataclass
class Landmarks:
    names: Tuple[str]
    coords: _npt.NDArray

    Self = 'Landmarks'

    @property
    def x(self) -> _npt.NDArray:
        return self.coords[:, 0]

    @property
    def y(self) -> _npt.NDArray:
        return self.coords[:, 1]

    @property
    def p(self) -> _npt.NDArray:
        return self.coords[:, 2]
    
    @property
    def xy(self) -> _npt.NDArray:
        """returns the x/y coordinates as a numpy array."""
        return self.coords[:, :2]
    
    def __len__(self):
        return len(self.names)

    def __iter__(self):
        for i, name in enumerate(self.names):
            yield Landmark(
                name=name,
                coords=self.coords[i, :].ravel()
            )

    def __getitem__(
        self,
        key: Union[str, int, slice, _npt.NDArray]
    ) -> Union[Self, Landmark]:
        if isinstance(key, str):
            if key not in self.names:
                raise KeyError(key)
            idx = self.names.index(key)
            return Landmark(
                name=self.names[idx],
                coords=self.coords[idx, :].ravel()
            )
        elif isinstance(key, int):
            return Landmark(
                name=self.names[key],
                coords=self.coords[key, :].ravel()
            )
        elif isinstance(key, slice):
            return self.__class__(
                name=self.names[key],
                coords=self.coords[key, :]
            )
        else:
            # assumes npt.NDArray
            raise NotImplementedError(f'TODO: Landmarks.__getitem__({key.__class__.__name__})')

    def ordered(self, keys: Iterable[str]) -> Self:
        """re-order the landmarks in accordance with the series of landmark names
        specified as ``keys``."""
        landmarks = [self[key] for key in keys if key in self.names]
        if len(landmarks) == 0:
            return self.__class__(names=(), coords=_np.array([]))
        else:
            return self.__class__(
                names=tuple(item.name for item in landmarks),
                coords=_np.stack([item.coords for item in landmarks], axis=0)
            )

    def affine_warp(self, warp: _npt.NDArray) -> Self:
        if len(self) == 0:
            return self
        coords = self.coords.copy()
        coords[:, :2] = _affine.warp_points(self.coords[:, :2], warp)
        return self.__class__(
            names=self.names,
            coords=coords
        )

    def annotate_image(
        self,
        image: _npt.NDArray,
        markersize: int = 16
    ) -> _npt.NDArray:
        buf  = _PILImage.fromarray(image)
        draw = _PILDraw.Draw(buf)
        rad  = markersize / 2
        for pt in self.points:
            if not pt.is_valid():
                continue
            draw.ellipse(
                [(round(pt.x - rad), round(pt.y - rad)),
                 (round(pt.x + rad), round(pt.y + rad))],
                 fill=(255, 255, 255)  # white
            )
        return _np.asarray(buf)

    @property
    def points(self) -> Tuple[Landmark]:
        return tuple(
            Landmark(self.names[i], self.coords[i,:]) for i in range(len(self.names))
        )

    @property
    def left(self) -> Self:
        return self.ordered(_refs.LEFT_LANDMARK_NAMES + _refs.MIDDLE_LANDMARK_NAMES)
    
    @property
    def right(self) -> Self:
        return self.ordered(_refs.MIDDLE_LANDMARK_NAMES + _refs.RIGHT_LANDMARK_NAMES)

    @property
    def middle(self) -> Self:
        return self.ordered(_refs.MIDDLE_LANDMARK_NAMES)

    @property
    def without_middle(self) -> Self:
        names = tuple(name for name in self.names if name not in _refs.MIDDLE_LANDMARK_NAMES)
        return self.ordered(names)
    
    @classmethod
    def from_single_landmarks(cls, landmarks: Iterable[Landmark]) -> Self:
        return cls(
            names=tuple(item.name for item in landmarks),
            coords=_np.stack([item.coords for item in landmarks], axis=0)
        )


@_dataclasses.dataclass
class Alignment:
    """holds the warp matrices for the left and the right hemispheres.
    By default, it is supposed to hold the transformation in the
    `reference --> image` direction."""
    left: _npt.NDArray  # warp matrix for the left hemisphere
    right: _npt.NDArray  # warp matrix for the right hemisphere

    separate: bool = True
    # indication of whether the left and
    # the right hemispheres were estimated separately

    Self = 'Alignment'

    def invert(self) -> Self:
        """returns the inverse warp matrices.
        currently, this method is only supported for Alignment objects
        with `separate = False`"""
        if self.separate != False:
            raise ValueError('inverting separate-hemispheres alignment is not supported' 
                             'use `separate_sides=True` when performing alignment')
        inv = _affine.invert(self.left)
        return self.__class__(
            left=inv,
            right=inv,
            separate=False
        )

    def warp_points(self, landmarks: Landmarks) -> Landmarks:
        left: Landmarks   = landmarks.left.without_middle
        middle: Landmarks = landmarks.middle
        right: Landmarks  = landmarks.right.without_middle

        left  = left.affine_warp(self.left)
        right = right.affine_warp(self.right)
        if self.separate is True:
            mleft  = middle.affine_warp(self.left)
            mright = middle.affine_warp(self.right)

            # NOTE: this is a temporary solution
            middle = Landmarks(
                names=middle.names,
                coords=(mleft.coords + mright.coords) / 2
            )
        else:
            middle = middle.affine_warp(self.left)
        return Landmarks.from_single_landmarks(
            left.points + middle.points + right.points
        )
    
    def warp_image(
        self,
        image: _npt.NDArray,
        side: Optional[Hemisphere] = 'left'
    ) -> _npt.NDArray:
        return _cv2.warpAffine(
            image,
            getattr(self, side),
            dsize=_defaults.VIDEO_FRAME_SIZE
        )
    
    def to_dict(self) -> Dict[str, float]:
        def update_(
            dct: Dict[str, float],
            side: str,
            warp: _npt.NDArray
        ):
            dct[f'{side}_xx'] = float(warp[0, 0])
            dct[f'{side}_xy'] = float(warp[0, 1])
            dct[f'{side}_xc'] = float(warp[0, 2])
            dct[f'{side}_yx'] = float(warp[1, 0])
            dct[f'{side}_yy'] = float(warp[1, 1])
            dct[f'{side}_yc'] = float(warp[1, 2])
        
        ret = dict(is_separate=bool(self.separate))
        update_(ret, 'left', self.left)
        update_(ret, 'right', self.right)
        return ret
    
    @classmethod
    def from_dict(cls, dct) -> Self:
        def reconstruct_(dct, side: str) -> _npt.NDArray:
            ret = _np.zeros((2, 3), dtype=_np.float32)
            for i, row in enumerate(('x', 'y')):
                for j, col in enumerate(('x', 'y', 'c')):
                    ret[i, j] = dct[f'{side}_{row}{col}']
            return ret

        separate = dct['is_separate']
        left     = reconstruct_(dct, 'left')
        right    = reconstruct_(dct, 'right')
        return Alignment(left=left, right=right, separate=separate)


def write_labeled_video(
    outpath: PathLike,
    dlc_output: DLCOutput,
    fps: Optional[Number] = None
):
    outpath = Path(outpath)
    if fps is None:
        fps = _defaults.VIDEO_FRAME_RATE
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)
    _iio.imwrite(str(outpath), dlc_output.images, fps=fps)


def write_dlc_landmarks(
    outpath: PathLike,
    dlc_output: DLCOutput
):
    outpath = Path(outpath)
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)
    dlc_output.table.to_csv(str(outpath), index=False, header=True)
    

def load_reference_landmarks() -> Landmarks:
    return Landmarks(
        names=tuple(_refs.LANDMARK_NAMES[i] for i in _refs.LANDMARK_IDS),
        coords=_np.array([_refs.LANDMARK_COORDS_512[item] + (1,) for item in _refs.LANDMARK_IDS])
    )  # (1,) is added to coords to represent `likelihood = 1.0`


def landmarks_from_dlc_output(
    output: Union[DLCOutput, _pd.DataFrame]
) -> Tuple[Landmarks]:
    if isinstance(output, DLCOutput):
        tab = output.table
    else:
        tab = output
    tab = tab.copy()
    tab.columns = tab.columns.droplevel()  # drops level 0 (i.e. scorer)

    frames = []
    for rowidx in range(tab.shape[0]):
        row = tab.iloc[rowidx].to_dict()
        coords = []
        for name in _refs.LANDMARK_IDS:
            name_ = name.lower()
            coords.append((row[name_, 'x'], row[name_, 'y'], row[name_, 'likelihood']))
        frames.append(Landmarks(
            names=tuple(_refs.LANDMARK_NAMES[i] for i in _refs.LANDMARK_IDS),
            coords=_np.array(coords)
        ))
    return tuple(frames)


def write_alignment_table(
    outpath: PathLike,
    alignment: Iterable[Alignment]
):
    outpath = Path(outpath)
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)
    tab = _pd.DataFrame([item.to_dict() for item in alignment])
    tab.to_csv(str(outpath), index=False, header=True)


def load_alignment_table(srcpath: PathLike) -> Tuple[Alignment]:
    tab = _pd.read_csv(srcpath)
    return tuple(
        Alignment.from_dict(tab.iloc[i].to_dict()) for i in range(tab.shape[0])
    )
