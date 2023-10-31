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
from typing import Optional, Union, Iterable, Tuple
import dataclasses

import numpy as _np
import numpy.typing as _npt
import imageio as _iio
import pandas as _pd
import cv2 as _cv2

from . import defaults as _defaults
from .typing import PathLike, Suffixes, Number
from mesonet.utils import natural_sort_key as _natural_sort_key  # FIXME


InputImageFiles = Tuple[Path]


@dataclasses.dataclass
class InputImages:
    pages: Tuple[int]
    images: Tuple[_npt.NDArray]
    shapes: Optional[_npt.NDArray] = None
    image_indexer: Optional[_npt.NDArray] = None
    page_indexer: Optional[_npt.NDArray] = None

    Self = 'InputImages'

    def __post_init__(self):
        size = sum(self.pages)
        self.shapes = _np.zeros((len(self.images), 2), dtype=_np.int_)
        self.image_indexer = _np.zeros(size, dtype=_np.int_)
        self.page_indexer  = _np.zeros(size, dtype=_np.int_)
        start = 0
        for image_idx, (width, img) in enumerate(zip(self.pages, self.images)):
            window = slice(start, start + width)
            if width == 1:
                self.shapes[image_idx, :] = img.shape
            else:
                self.shapes[image_idx, :] = img.shape[1:]
            self.image_indexer[window] = image_idx
            self.page_indexer[window] = _np.arange(width)
            start += width

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError(f'InputImages only accepts int indices, got {type(idx)}')
        image_idx = self.image_indexer[idx]
        if self.pages[image_idx] > 1:
            page_idx  = self.page_indexer[idx]
            return self.images[image_idx][page_idx]
        else:
            return self.images[image_idx]

    def __iter__(self):
        for pages, image in zip(self.pages, self.images):
            if pages == 1:
                yield image
            else:
                for page in range(pages):
                    yield image[page]

    @property
    def widths(self) -> _npt.NDArray:
        return self.shapes[:, 1]

    @property
    def heights(self) -> _npt.NDArray:
        return self.shapes[:, 0]

    @classmethod
    def from_grayscale_images(cls, images: Iterable[_npt.NDArray]) -> Self:
        return cls(pages=tuple(img.shape[3] for img in images), images=tuple(images))


def collect_image_files(input_dir: PathLike, suffixes: Optional[Suffixes] = None) -> InputImageFiles:
    if suffixes is None:
        suffixes = _defaults.IMAGE_SUFFIXES
    if isinstance(suffixes, str):
        suffixes = (suffixes, )
    input_dir = Path(input_dir)
    files = []
    for suffix in suffixes:
        for child in input_dir.glob(f'*{suffix}'):
            files.append(child)
    return tuple(sorted(files, key=lambda file: _natural_sort_key(str(file))))


def load_images(
    input_files: Union[InputImageFiles, PathLike],
    suffixes: Optional[Suffixes] = None
) -> InputImages:
    """``suffixes`` will only be used if ``input_files`` is a path-like object."""
    if isinstance(input_files, (str, Path)):
        input_files = collect_image_files(input_files, suffixes=suffixes)

    def load_single(path: Path) -> Tuple[_npt.NDArray, int]:
        img = _iio.v3.imread(str(path))
        if img.shape[-1] == 3:
            # probably in RGB
            dtype = img.dtype
            img = img.mean(-3).astype(dtype)

        if path.suffix.startswith('.tif'):
            # heuristic parsing of the image number
            shape = img.shape
            if len(shape) == 2:
                # single-page mode
                num = 1
            elif len(shape) == 3:
                # multi-page mode
                num = shape[0]
                if num == 1:
                    img = _np.squeeze(img)
            else:
                raise ValueError(f'unexpected image shape {shape} found from: {path.name}')
            return img, num
        else:
            return img, 1

    pages  = []
    images = []
    for srcpath in input_files:
        img, num = load_single(srcpath)
        images.append(img)
        pages.append(num)
    return InputImages(pages=tuple(pages), images=tuple(images))


def write_rescaled_video(
    outpath: PathLike,
    images: InputImages,
    fps: Optional[Number] = None
):
    if fps is None:
        fps = _defaults.VIDEO_FRAME_RATE
    outpath = Path(outpath)
    rescaled = []
    for img, pagenum in zip(images.images, images.pages):
        if pagenum > 1:
            for page in range(pagenum):
                rimg = _cv2.resize(img[page], _defaults.VIDEO_FRAME_SIZE,
                                   interpolation=_cv2.INTER_LINEAR)
                rescaled.append(
                    _np.stack([rimg, rimg, rimg], axis=-1)
                )
        else:
            rimg = _cv2.resize(img, _defaults.VIDEO_FRAME_SIZE,
                               interpolation=_cv2.INTER_LINEAR)
            rescaled.append(
                _np.stack([rimg, rimg, rimg], axis=-1)
            )
    rescaled = _np.stack(rescaled, axis=0)

    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)
    _iio.v3.imwrite(str(outpath), rescaled, fps=fps)


def write_metadata_table(
    outpath: PathLike,
    images: InputImages,
    image_names: Optional[Iterable[str]] = None
):
    outpath = Path(outpath)
    image_names = tuple(image_names)

    data = []
    if image_names is not None:
        if len(image_names) != len(images):
            raise ValueError(
                f"the length of the image names ({len(image_names)}) "
                f"does not match the length of the images ({len(images)})"
            )
    
    widths = images.widths
    heights = images.heights
    pages = images.pages
    for idx in range(len(images)):
        for page in range(pages[idx]):
            _data = {}
            if image_names is not None:
                _data['Image'] = image_names[idx]
            _data['Frame'] = page + 1
            _data['Width'] = widths[idx]
            _data['Height'] = heights[idx]
            _data['TotalFrames'] = pages[idx]
            data.append(_data)
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)
    _pd.DataFrame(data).to_csv(str(outpath), header=True, index=(image_names is None))


def load_metadata_table(
    path: PathLike
) -> _pd.DataFrame:
    return _pd.read_csv(str(path))


def collected_images_video_path(output_dir: PathLike) -> Path:
    return Path(output_dir) / _defaults.COLLECTED_IMAGES_VIDEO_NAME


def collected_images_metadata_path(output_dir: PathLike) -> Path:
    return Path(output_dir) / _defaults.COLLECTED_IMAGES_METADATA_NAME
