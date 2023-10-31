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

from typing import Optional
from pathlib import Path
import tempfile as _tempfile
import shutil as _shutil

import pandas as _pd
import imageio.v3 as _iio

from ..typing import (
    PathLike,
)
from . import (
    base as _base,
    paths as _paths,
)


def predict_dlc_landmarks(
    video_path: PathLike,
    dlc_project_dir: Optional[PathLike] = None
) -> _base.DLCOutput:
    """use the video file at ``video_path``
    to estimate landmarks using DeepLabCut.

    ``dlc_project_dir`` is supposed to point to the
    DeepLabCut project directory to be used.
    If None is provided here, the value specified
    in the ``MESONET_DLC_PROJECT_DIR`` environment
    variable will be used.
    """
    video_path = Path(video_path)
    configpath = _paths.dlc_config_path(dlc_project_dir)

    # run DLC inside the temp dir
    tempdir = Path(_tempfile.mkdtemp(prefix='temp', dir=str(video_path.parent)))
    try:
        import deeplabcut as _dlc
        import os as _os

        _cwd = _os.getcwd()  # NOTE: just in case DLC calls `chdir`

        # copy the video under the working dir
        # NOTE: DLC seems to have issues resolving relative paths
        sourcepath = str(_shutil.copy(video_path, tempdir / video_path.name).absolute())
        _dlc.analyze_videos(
            str(configpath),
            [sourcepath],
            videotype='.mp4'
        )
        _os.chdir(str(_cwd))  # NOTE: just in case DLC calls `chdir`
        _dlc.create_labeled_video(
            str(configpath),
            [sourcepath],
            filtered=False
        )
        _os.chdir(str(_cwd))  # NOTE: just in case DLC calls `chdir`

        tablepath = search_pattern(tempdir, '*.h5')
        labeledpath = search_pattern(tempdir, '*labeled.mp4')

        table = _pd.read_hdf(tablepath, key='df_with_missing')
        labeled = _iio.imread(str(labeledpath))
        return _base.DLCOutput(table=table, images=labeled)
    finally:
        _shutil.rmtree(tempdir)


def search_pattern(directory: Path, pattern: str) -> Path:
    candidates = list(directory.glob(pattern))
    if len(candidates) == 0:
        raise FileNotFoundError(f"no candidates with pattern '{pattern}' found in: {directory}")
    elif len(candidates) > 1:
        raise ValueError(f"multiple candidates with pattern '{pattern}' found in: {directory}")
    return candidates[0]
