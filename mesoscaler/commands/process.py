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

from typing import Optional, Iterable
from pathlib import Path
import tempfile as _tempfile

from ..typing import (
    ResultsFileType,
)
from .. import (
    defaults as _defaults,
    procs as _procs,
)
from . import (
    validate as _validate,
)
from .root import commands as _commands


def run(
    image_paths: Iterable[str],
    output_dir: Optional[str] = None,
    output_file_type: ResultsFileType = 'hdf',
    dlc_project_dir: Optional[str] = None,
    threshold: Optional[float] = None,
    min_valid_points: Optional[int] = None,  # TODO: this may be configurable
    separate_sides: Optional[bool] = False,  # this value is intentionally made un-configurable
    video_fps: Optional[float] = None,
    rois_file_type: ResultsFileType = 'hdf',  # this value is intentionally made un-configurable
    resize_rois: bool = True,  # this value is intentionally made un-configurable
):
    image_paths = _validate.image_paths(image_paths)
    output_dir  = _validate.output_directory(output_dir)
    output_file_type = _validate.output_file_type(output_file_type)
    if (image_paths is None) or (output_dir is None):
        return
    if (output_file_type is None):
        return
    process_dir = Path(_tempfile.mkdtemp(prefix='temp-ksm-process'))
    collect_dir = process_dir / 'collected'
    landmarks_dir = process_dir / 'landmarks'
    rois_dir = process_dir / 'rois'
    # TODO:
    # better wrapping the following procedures
    # using a try block so that the error message becomes minimal?
    try:
        _procs.run_image_collection(
            image_paths,
            output_dir=collect_dir,
            video_fps=video_fps
        )
        _procs.run_landmark_prediction(
            collect_dir,
            landmarks_dir,
            dlc_project_dir=dlc_project_dir,
            video_fps=video_fps
        )
        _procs.run_landmark_alignment(
            landmarks_dir,
            landmarks_dir,
            likelihood_threshold=threshold,
            min_valid_points=min_valid_points,
            separate_sides=separate_sides,
            video_fps=video_fps,
        )
        _procs.run_rois_generation(
            collect_dir,
            landmarks_dir,
            rois_dir,
            file_type=rois_file_type,
            resize=resize_rois,
        )
        _procs.run_packaging_all_results(
            metadata_dir=collect_dir,
            landmarks_dir=landmarks_dir,
            alignment_dir=landmarks_dir,
            rois_dir=rois_dir,
            output_dir=output_dir,
            filetype=output_file_type
        )
    finally:
        import shutil
        shutil.rmtree(process_dir)


parser = _commands.add_parser(
    'process',
    help='end-to-end processing of the given set of images.'
)
parser.add_argument(
    '-F',
    '--output-file-type',
    dest='output_file_type',
    metavar='TYPE',
    choices=['hdf', 'matlab'],
    help=f"the output file type to be generated. defaults to '{_defaults.PACKAGE_FILE_TYPE}'"
)
parser.add_argument(
    '-P',
    '--dlc-project-dir',
    dest='dlc_project_dir',
    metavar='PROJECT-DIR',
    help='the MesoNet DeepLabCut project directory to be used for landmark prediction. If not supplied, it tries to read from the `MESONET_DLC_PROJECT_DIR` environment variable.'
)
parser.add_argument(
    '-T',
    '--likelihood-threshold',
    dest='threshold',
    metavar='THRESHOLD-VALUE',
    type=float,
    help=f'the bound in the likelihood of landmark prediction to be used in alignment (defaults to {_defaults.LANDMARK_LIKELIHOOD_THRESHOLD}).'
)
parser.add_argument(
    '--video-fps',
    dest='video_fps',
    metavar='FPS',
    type=float,
    help=f'the frame rate of the output images video (defaults to {_defaults.VIDEO_FRAME_RATE}).'
)
parser.add_argument(
    '-o',
    '--output-directory',
    dest='output_dir',
    metavar='OUTPUT-DIR',
    help='the path of the output directory to store the outputs (defaults to the current directory).'
)
parser.add_argument(
    'image_paths',
    nargs='+',
    metavar='IMAGE-PATH',
    help='the path(s) to the image files to be processed.'
)
parser.set_defaults(
    func=run,
    separate_sides=False,
    resize_rois=True,
    rois_file_type='hdf',
    output_file_type='hdf'
)
