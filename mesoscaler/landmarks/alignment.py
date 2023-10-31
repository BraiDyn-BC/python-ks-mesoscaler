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

from typing import Union, Optional, Iterable, Tuple, Generator
import dataclasses as _dataclasses
import math as _math

import numpy.typing as _npt

from ..typing import (
    Number,
)
from .. import (
    defaults as _defaults,
)
from . import (
    reference as _refs,
    base as _base,
    affine as _affine,
)


@_dataclasses.dataclass
class Pairing:
    """holds a set of landmarks for a single target image (512 x 512) and the reference frame."""
    target: _base.Landmarks
    reference: _base.Landmarks

    Self = 'Pairing'

    def __len__(self):
        return len(self.target)

    def __iter__(self) -> Generator[Tuple[_base.Landmark, _base.Landmark], None, None]:
        yield from zip(self.target, self.reference)

    def ordered(self, keys: Iterable[str]) -> Self:
        return self.__class__(
            target=self.target.ordered(keys),
            reference=self.reference.ordered(keys)
        )

    @property
    def names(self) -> Tuple[str]:
        return self.target.names

    @property
    def left(self) -> Self:
        return self.ordered(_refs.LEFT_LANDMARK_NAMES + _refs.MIDDLE_LANDMARK_NAMES)

    @property
    def right(self) -> Self:
        return self.ordered(_refs.RIGHT_LANDMARK_NAMES + _refs.MIDDLE_LANDMARK_NAMES)

    @property
    def middle(self) -> Self:
        return self.ordered(_refs.MIDDLE_LANDMARK_NAMES)

    @property
    def without_middle(self) -> Self:
        names = tuple(name for name in self.names if name not in _refs.MIDDLE_LANDMARK_NAMES)
        return self.ordered(names)


def align_dlc_landmarks(
    target: Union[_base.DLCOutput, _base.Landmarks, Iterable[_base.Landmarks]],
    reference: Optional[_base.Landmarks],
    likelihood_threshold: Optional[Number] = None,
    separate_sides: Optional[bool] = False
) -> Tuple[_base.Alignment]:
    """takes `Landmark` objects, and try to estimate the Affine warp matrix
    that warps `reference` landmarks to `target` landmarks.
    returns an `Alignment` object that holds the estimated Affine matrix (matrices).

    The target landmarks (those taken from the actual data) will be validated
    based on its prediction likelihood, using the `likelihood_threshold` value.

    `target` can be a `DLCOutput` object (internally uses `landmarks_from_dlc_output`
    to convert).

    If `reference` is not supplied, `load_reference_landmarks` will be used to
    obtain the reference landmarks.

    Internally uses `validate_landmarks` and `estimate_warp_matrix`.
    """
    if isinstance(target, _base.DLCOutput):
        target = _base.landmarks_from_dlc_output(target)
    elif isinstance(target, _base.Landmarks):
        target = (target,)
    else:
        target = tuple(target)
    if reference is None:
        reference = _base.load_reference_landmarks()
    return tuple(
        estimate_warp_matrix(
            validate_landmarks(
                single,
                reference,
                likelihood_threshold=likelihood_threshold
            ),
            separate_sides=separate_sides
        ) for single in target
    )


def validate_landmarks(
    target: _base.Landmarks,
    reference: _base.Landmarks,
    likelihood_threshold: Optional[Number] = None
) -> Pairing:
    """cherry-picks only the 'valid' landmarks out of `target` and `reference`
    `Landmarks` object, using the `likelihood_threhold` criterion.
    
    returns a `Pairing` object (which is merely a set of `reference` and `target`,
    but with some utility methods)
    """
    if likelihood_threshold is None:
        likelihood_threshold = _defaults.LANDMARK_LIKELIHOOD_THRESHOLD
    valid_targets = []
    valid_refs = []
    for tg, ref in zip(target, reference):
        if tg.p > likelihood_threshold:
            valid_targets.append(tg)
            valid_refs.append(ref)
    return Pairing(
        target=_base.Landmarks.from_single_landmarks(valid_targets),
        reference=_base.Landmarks.from_single_landmarks(valid_refs)
    )


def estimate_warp_matrix(
    validated_pair: Pairing,
    separate_sides: Optional[bool] = None
) -> _base.Alignment:
    """takes a `Pairing` object and estimates the Affine warp matrix (matrices)
    corresponding to the reference-target pairs of landmarks.
    
    The paired landmarks are assumed to have been 'validated' by the user,
    e.g. by means of `validate_landmarks`.

    Internally uses the `estimate_affine` function.
    """
    def _align(pair: Pairing) -> _npt.NDArray:
        return _affine.estimate(
            pair.reference.xy,
            pair.target.xy
        )

    left: Pairing = validated_pair.left
    right: Pairing = validated_pair.right

    if separate_sides is None:
        separate_sides = (len(left.without_middle) > 2) and (len(right.without_middle) > 2)

    if separate_sides is True:
        return _base.Alignment(
            left=_align(left),
            right=_align(right),
            separate=True
        )
    else:
        warp = _align(validated_pair)
        return _base.Alignment(
            left=warp,
            right=warp,
            separate=False
        )


def update_dlc_landmarks(
    orig: _base.DLCOutput,
    reference: Optional[_base.Landmarks] = None,
    alignment: Optional[Iterable[_base.Alignment]] = None
) -> _base.DLCOutput:
    """uses the set of `alignment` to warp `reference` points
    and replace them with those in `orig` table.
    
    for how to obtain `Alignment`, 
    also see: `align_dlc_landmarks`
    """
    if reference is None:
        reference = _base.load_reference_landmarks()
    if alignment is not None:
        landmarks = [
            align.warp_points(reference) for align in alignment
        ]
    else:
        landmarks = [reference] * orig.size

    # update table
    newtable  = orig.table.copy()
    scorer    = newtable.columns[0][0]
    pointIDs = set(col[1] for col in newtable.columns)

    for pointID in pointIDs:
        pointname = _refs.LANDMARK_NAMES[pointID.upper()]
        for rowidx, marks in enumerate(landmarks):
            if pointname in marks.names:
                mark = marks[pointname]
                newtable.loc[rowidx, (scorer, pointID, 'x')] = mark.x
                newtable.loc[rowidx, (scorer, pointID, 'y')] = mark.y
            else:
                newtable.loc[rowidx, (scorer, pointID, 'x')] = _math.nan
                newtable.loc[rowidx, (scorer, pointID, 'y')] = _math.nan

    # update images
    images = orig.images
    if images is not None:
        images = images.copy()
        for i, marks in enumerate(landmarks):
            images[i] = marks.annotate_image(images[i])

    return _base.DLCOutput(table=newtable, images=images)
