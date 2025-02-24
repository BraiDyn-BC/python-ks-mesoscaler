# MIT License
#
# Copyright (c) 2023-2024 Keisuke Sehara
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

from importlib import reload as _reload  # DEBUG

from . import (  # noqa: F401
    validate,
    root,
    process,
    image_collection_step,
    landmark_prediction_step,
    atlas_alignment_step,
    roi_generation_step,
    packaging_step,
)

# DEBUG
_reload(validate)
_reload(root)
_reload(image_collection_step)
_reload(landmark_prediction_step)
_reload(atlas_alignment_step)
_reload(roi_generation_step)
_reload(packaging_step)
_reload(process)


parser = root.parser


def parse(*args):
    parsed = vars(parser.parse_args(args))
    fn = vars.pop('func')
    fn(**parsed)


def run():
    parsed = vars(parser.parse_args())
    fn = parsed.pop('func')
    fn(**parsed)
