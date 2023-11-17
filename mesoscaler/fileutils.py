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

from typing import Optional, Iterable, Tuple, Union # noqa: F401
from pathlib import Path
import math as _math
import re as _re

from .typing import (
    ResultsFileType,
)


def index_name(text: str) -> Tuple[Union[str, int]]:
    """used for sorting file names"""
    digits = _re.compile("([0-9]+)")
    return tuple(int(item) if item.isdigit() else item.lower() for item in digits.split(text))


def unique_names_from_path(paths: Iterable[Path]) -> Tuple[str]:
    NUM_MAX_TEST = 10

    def with_level(i):
        return [str(path.relative_to(path.parents[i])) for path in paths]

    def is_unique(names):
        return len(set(names)) == len(names)

    for i in range(NUM_MAX_TEST):
        names = with_level(i)
        if is_unique(names):
            return tuple(names)
    raise RuntimeError('failed to find unique names from a set of paths')


def required_number_of_digits(total: int, minimum: int = 2) -> int:
    required = _math.floor(_math.log10(total)) + 1
    return max(required, minimum)


def get_roi_file_suffix(filetype: ResultsFileType) -> str:
    if filetype == 'hdf':
        return '.h5'
    elif filetype == 'matlab':
        return '.mat'
    else:
        raise ValueError(f"ROI file type expected to be one of ('hdf', 'matlab'), got {repr(filetype)}")
