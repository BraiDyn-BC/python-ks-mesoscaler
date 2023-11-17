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
from typing import Optional
import os as _os

from .. import (
    defaults as _defaults,
)
from ..typing import (
    PathLike,
)

ENV_MESONET_DLC_DIR = 'MESONET_DLC_PROJECT_DIR'
DLC_CONFIG_FILE_NAME = 'config.yaml'


def dlc_config_path(project_dir: Optional[Path] = None) -> Path:
    project_dir = dlc_project_dir(project_dir)
    config = project_dir / DLC_CONFIG_FILE_NAME
    if not config.exists():
        raise FileNotFoundError(
            "the directory does not seem to be a DeepLabCut project: "
            f"{project_dir}"
        )
    return config


def dlc_project_dir(directory: Optional[PathLike] = None) -> Path:
    if directory is None:
        if ENV_MESONET_DLC_DIR not in _os.environ.keys():
            raise KeyError(
                f"Please specify the '{ENV_MESONET_DLC_DIR}' environment variable "
                "to point to the DeepLabCut project directory to be used."
            )
        return Path(_os.environ[ENV_MESONET_DLC_DIR])
    else:
        return Path(directory)


def predicted_landmarks_video_path(output_dir: Path) -> Path:
    return output_dir / _defaults.PREDICTED_LANDMARKS_VIDEO_NAME


def predicted_landmarks_table_path(output_dir: Path) -> Path:
    return output_dir / _defaults.PREDICTED_LANDMARKS_TABLE_NAME


def alignment_table_path(output_dir: Path) -> Path:
    return output_dir / _defaults.ALIGNMENT_TABLE_NAME


def aligned_landmarks_video_path(output_dir: Path) -> Path:
    return output_dir / _defaults.ALIGNED_LANDMARKS_VIDEO_NAME


def aligned_landmarks_table_path(output_dir: Path) -> Path:
    return output_dir / _defaults.ALIGNED_LANDMARKS_TABLE_NAME
