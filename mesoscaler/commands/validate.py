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
import sys as _sys


def image_paths(image_paths: Iterable[str]) -> Optional[Iterable[Path]]:
    image_paths = [Path(path) for path in image_paths]
    if len(image_paths) == 0:
        return _abort('at least one image file must be supplied')
    for path in image_paths:
        if not path.exists():
            return _abort(f'file not accessible: {str(path)}')
    return image_paths


def input_directory(inputdir: Optional[str] = None) -> Optional[Path]:
    inputdir = Path(inputdir) if inputdir is not None else Path()
    if not inputdir.exists():
        return _abort(f'input directory not accessible: {str(inputdir)}')
    return inputdir


def output_directory(outdir: Optional[str] = None) -> Optional[Path]:
    outdir = Path(outdir) if outdir is not None else Path()
    return outdir


def output_file_type(file_type: str) -> Optional[str]:
    if file_type in ('hdf', 'matlab'):
        return file_type
    else:
        return _abort(f"unknown output file type: '{file_type}'")


def _abort(msg: str):
    print(f"***{msg}", file=_sys.stderr, flush=True)
    print(_sys.argv)
