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

from .. import (
    defaults as _defaults,
    procs as _procs,
)
from . import (
    validate as _validate,
)
from .root import commands as _commands

def run(
    input_directory: Optional[str] = None,
    outdir: Optional[str] = None,
    dlc_project_dir: Optional[str] = None,
    video_fps: Optional[float] = None
):
    indir = _validate.input_directory(input_directory)
    outdir = _validate.output_directory(outdir)
    if (indir is None) or (outdir is None):
        return
    # TODO: make sure that input_directory
    # correctly contains the metadata and the video files
    _procs.run_landmark_prediction(
        indir,
        outdir,
        dlc_project_dir=dlc_project_dir,
        video_fps=video_fps
    )

parser = _commands.add_parser(
    'landmark-prediction-step',
    help='(step 2) predicts reference landmarks from the (collected/packed) input images.'
)
parser.add_argument(
    '--video-fps',
    dest='video_fps',
    metavar='FPS',
    type=float,
    help=f'the frame rate of the output images video (defaults to {_defaults.VIDEO_FRAME_RATE}).'
)
parser.add_argument(
    '-P',
    '--dlc-project-dir',
    dest='dlc_project_dir',
    metavar='PROJECT-DIR',
    help='the MesoNet DeepLabCut project directory to be used for landmark prediction. If not supplied, it tries to read from the `MESONET_DLC_PROJECT_DIR` environment variable.'
)
parser.add_argument(
    '-o',
    '--output-directory',
    dest='outdir',
    metavar='OUTPUT-DIRECTORY',
    help='the path of the output directory to store predicted landmarks data (defaults to the current directory).'
)
parser.add_argument(
    'input_directory',
    nargs='?',
    default=None,
    metavar='INPUT-DIR',
    help='the path to the (collected/packed) images directory to be processed (tries to use the current directory if nothing is supplied).'
)
parser.set_defaults(func=run)
