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
    outdir: Optional[str] = None,
    video_fps: Optional[float] = None
):
    image_paths = _validate.image_paths(image_paths)
    outdir      = _validate.output_directory(outdir)
    if (image_paths is None) or (outdir is None):
        return
    _procs.run_image_collection(
        image_paths,
        output_dir=outdir,
        video_fps=video_fps,
    )


parser = _commands.add_parser(
    'image-collection-step',
    help='(step 1) collects a set of images, and rescales/packs them to be processed.'
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
    help='the path of the output directory to store metadata and packed images data (defaults to the current directory).'
)
parser.add_argument(
    'image_paths',
    nargs='+',
    metavar='IMAGE-PATH',
    help='the path(s) to the image files to be processed.'
)
parser.set_defaults(func=run)
