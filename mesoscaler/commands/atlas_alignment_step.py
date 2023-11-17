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
    threshold: Optional[float] = None,
    separate_sides: Optional[bool] = False,  # this value is intentionally made un-configurable
    video_fps: Optional[float] = None,
):
    indir = _validate.input_directory(input_directory)
    outdir = _validate.output_directory(outdir)
    if (indir is None) or (outdir is None):
        return
    # TODO:
    # make sure that the input directory contains
    # the landmark prediction data
    _procs.run_landmark_alignment(
        indir,
        outdir,
        likelihood_threshold=threshold,
        separate_sides=separate_sides,
        video_fps=video_fps,
    )


parser = _commands.add_parser(
    'atlas-alignment-step',
    help='(step 3) aligns the reference atlas to the input images based on the predicted landmarks.'
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
    dest='outdir',
    metavar='OUTPUT-DIRECTORY',
    help='the path of the output directory to store alignment information (defaults to the current directory).'
)
parser.add_argument(
    'input_directory',
    nargs='?',
    default=None,
    metavar='INPUT-DIR',
    help='the path to the directory containing the predicted landmarks (tries to use the current directory if nothing is supplied).'
)
parser.set_defaults(func=run, separate_sides=False)
