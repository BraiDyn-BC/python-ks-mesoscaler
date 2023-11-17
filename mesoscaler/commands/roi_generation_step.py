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

from ..typing import (
    ResultsFileType,
)
from .. import (
    procs as _procs,
)
from . import (
    validate as _validate,
)
from .root import commands as _commands


def run(
    input_directory: Optional[str] = None,
    metadir: Optional[str] = None,
    outdir: Optional[str] = None,
    resize: bool = True,  # this value is intentionally made un-configurable
    file_type: ResultsFileType = 'hdf',  # this value is intentionally made un-configurable
):
    aligndir = _validate.input_directory(input_directory)
    metadir  = _validate.input_directory(metadir)
    outdir   = _validate.output_directory(outdir)
    if (aligndir is None) or (metadir is None) or (outdir is None):
        return
    # TODO:
    # make sure that the file metadata
    # and the alignment metadata exist
    _procs.run_rois_generation(
        metadir,
        aligndir,
        outdir,
        file_type=file_type,
        resize=resize
    )


parser = _commands.add_parser(
    'roi-generation-step',
    help='(step 4) generate roi masks for the images based on the aligned reference atlas.'
)
parser.add_argument(
    '-m',
    '--metadata-directory',
    dest='metadir',
    metavar='METADATA-DIRECTORY',
    help='the path to the metadata directory (which contains files generated in the image-collection-step). If not supplied, it tries to find it from the current directory.'
)
parser.add_argument(
    '-o',
    '--output-directory',
    dest='outdir',
    metavar='OUTPUT-DIRECTORY',
    help='the path of the output directory to store the generated ROIs (defaults to the current directory).'
)
parser.add_argument(
    'input_directory',
    nargs='?',
    default=None,
    metavar='INPUT-DIR',
    help='the path to the directory containing the estimated alignment information (tries to use the current directory if nothing is supplied).'
)
parser.set_defaults(
    func=run,
    resize=True,
    file_type='hdf'
)
