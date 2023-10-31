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

"""the procedures that correspond to the individual steps of the pipeline."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as _pd

from . import (
    images as _images,
    landmarks as _landmarks,
    rois as _rois,
    misc as _misc,
)
from .typing import (
    PathLike,
    Suffixes,
    Number,
    ROIFileType,
)


def run_image_collection(
    input_dir: PathLike,
    output_dir: PathLike,
    suffixes: Optional[Suffixes] = None,
    video_fps: Optional[Number] = None
) -> Path:
    """collect images from ``input_dir``, and stores the followings
    in ``output_dir``. creates the directory if not existent.
    returns the output directory as a ``Path``."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    paths  = _images.collect_image_files(input_dir, suffixes=suffixes)
    images = _images.load_images(paths)
    image_names = _misc.unique_names_from_path(paths)

    video_path = _images.collected_images_video_path(output_dir)
    table_path = _images.collected_images_metadata_path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    _images.write_rescaled_video(video_path, images, fps=video_fps)
    _images.write_metadata_table(table_path, images, image_names=image_names)
    return output_dir


def run_landmark_prediction(
    input_dir: PathLike,
    output_dir: PathLike,
    dlc_project_dir: Optional[PathLike] = None,
    video_fps: Optional[Number] = None
) -> Path:
    """uses the collected images video from ``input_dir``, and predict landmarks
    using the DeepLabCut network.
    
    writes the labeled video and the prediction results into ``output_dir``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if dlc_project_dir is None:
        dlc_project_dir = _landmarks.paths.dlc_project_dir()
    else:
        dlc_project_dir = Path(dlc_project_dir)
    source_video = _images.collected_images_video_path(input_dir)
    output = _landmarks.predict_dlc_landmarks(source_video, dlc_project_dir)
    labeled_video_path = _landmarks.predicted_landmarks_video_path(output_dir)
    labels_table_path  = _landmarks.predicted_landmarks_table_path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    _landmarks.write_labeled_video(labeled_video_path, output, fps=video_fps)
    _landmarks.write_dlc_landmarks(labels_table_path, output)
    return output_dir


def run_landmark_alignment(
    input_dir: PathLike,
    output_dir: PathLike,
    likelihood_threshold: Optional[float] = None,
    separate_sides: bool = True,
    video_fps: Optional[Number] = None
) -> Path:
    """loads the results of DeepLabCut inference from ``input_dir``, and
    estimate Affine warp matrix (matrices) for warping reference landmarks
    onto the actual data images.
    
    Writes out the results to ``output_dir``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    dlc_in: _landmarks.DLCOutput = _landmarks.DLCOutput.from_directory(input_dir)
    landmarks: Tuple[_landmarks.Landmarks] = _landmarks.landmarks_from_dlc_output(dlc_in)
    reference: _landmarks.Landmarks        = _landmarks.load_reference_landmarks()
    alignment: Tuple[_landmarks.Alignment] = _landmarks.align_dlc_landmarks(
        landmarks,
        reference,
        likelihood_threshold=likelihood_threshold,
        separate_sides=separate_sides
    )
    alignment_table_path = _landmarks.alignment_table_path(output_dir)
    _landmarks.write_alignment_table(alignment_table_path, alignment)

    dlc_out    = _landmarks.update_dlc_landmarks(
        dlc_in,
        reference=reference,
        alignment=alignment
    )
    aligned_table_path = _landmarks.aligned_landmarks_table_path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    _landmarks.write_dlc_landmarks(aligned_table_path, dlc_out)

    if dlc_out.images is not None:
        aligned_video_path = _landmarks.aligned_landmarks_video_path(output_dir)
        _landmarks.write_labeled_video(aligned_video_path, dlc_out, fps=video_fps)
    return output_dir


def run_rois_generation(
    metadata_dir: PathLike,
    alignment_dir: PathLike,
    output_dir: PathLike,
    file_type: ROIFileType
) -> Path:
    metadata_dir = Path(metadata_dir)
    alignment_dir = Path(alignment_dir)
    output_dir = Path(output_dir)

    metadata_table_path = _images.collected_images_metadata_path(metadata_dir)
    metadata = _images.load_metadata_table(metadata_table_path)

    alignment_table_path = _landmarks.alignment_table_path(alignment_dir)
    alignment: Tuple[_landmarks.Alignment] = _landmarks.load_alignment_table(alignment_table_path)

    for roiset in _rois.generate_rois_batch(
        alignment,
        metadata=metadata,
        reference=_rois.load_reference_ROIs(),
        outline=_rois.load_reference_outlines(),
        resize=True
    ):
        if roiset.total_frames == 1:
            outbase = roiset.image_name
        else:
            digits = _misc.required_number_of_digits(roiset.total_frames)
            outbase = f"{Path(roiset.image_name).stem}_frame{str(roiset.frame_idx).zfill(digits)}"
        suffix = _misc.get_roi_file_suffix(file_type)
        outfile = output_dir / f"{outbase}{suffix}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        roiset.to_file(outfile, file_type)
    return output_dir
