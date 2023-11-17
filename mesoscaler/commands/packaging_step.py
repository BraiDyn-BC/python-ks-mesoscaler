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
    defaults as _defaults,
)
from . import (
    validate as _validate
)
from .root import commands as _commands

def run(
    metadata_dir: Optional[str] = None,
    landmarks_dir: Optional[str] = None,
    alignment_dir: Optional[str] = None,
    rois_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_file_type: ResultsFileType = 'hdf'
):
    metadata_dir = _validate.input_directory(metadata_dir)
    landmarks_dir = _validate.input_directory(landmarks_dir)
    alignment_dir = _validate.input_directory(alignment_dir)
    rois_dir = _validate.input_directory(rois_dir)
    output_dir = _validate.output_directory(output_dir)
    output_file_type = _validate.output_file_type(output_file_type)

    if (metadata_dir is None) or (landmarks_dir is None) or (alignment_dir is None) or (rois_dir is None) or (output_dir is None):
        return
    if (output_file_type is None):
        return
    
    # TODO:
    # check directories (with their contents)
    _procs.run_packaging_all_results(
        metadata_dir=metadata_dir,
        landmarks_dir=landmarks_dir,
        alignment_dir=alignment_dir,
        rois_dir=rois_dir,
        output_dir=output_dir,
        filetype=output_file_type
    )

parser = _commands.add_parser(
    'packaging-step',
    help='single-step 5: pack all the results into single files for each image.'
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
    '-L',
    '--landmarks-directory',
    dest='landmarks_dir',
    metavar='LANDMARKS-DIR',
    help='the path to the directory containing the predicted landmarks (tries to use the current directory if nothing is supplied).'
)
parser.add_argument(
    '-A',
    '--alignment-directory',
    dest='alignment_dir',
    metavar='ALIGNMENT-DIR',
    help='the path to the directory containing the estimated alignment information (tries to use the current directory if nothing is supplied).'
)
parser.add_argument(
    '-R',
    '--rois-directory',
    dest='rois_dir',
    metavar='ROIS-DIR',
    help='the path to the directory containing the generated ROIs (tries to use the current directory if nothing is supplied).'
)
parser.add_argument(
    '-o',
    '--output-directory',
    dest='output_dir',
    metavar='OUTPUT-DIR',
    help='the path of the output directory to store the packaged files (defaults to the current directory).'
)
parser.add_argument(
    'metadata_dir',
    nargs='?',
    default=None,
    metavar='METADATA-DIRECTORY',
    help='the path to the metadata directory (which contains files generated in the image-collection-step). If not supplied, it tries to find it from the current directory.'
)
parser.set_defaults(func=run, output_file_type='hdf')
