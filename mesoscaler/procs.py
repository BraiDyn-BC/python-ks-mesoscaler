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

"""the procedures that correspond to the individual steps of the pipeline."""

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as _pd
import imageio.v3 as _iio

from . import (
    images as _images,
    landmarks as _landmarks,
    rois as _rois,
    packaging as _packaging,
    fileutils as _fileutils,
)
from .typing import (
    PathLike,
    InputImageFiles,
    Number,
    Suffixes,
    ResultsFileType,
)


def run_image_collection(
    input_dir_or_files: Union[PathLike, InputImageFiles],
    output_dir: PathLike,
    suffixes: Optional[Suffixes] = None,
    video_fps: Optional[Number] = None
) -> Path:
    """collect images from ``input_dir_or_files``, and stores the followings
    in ``output_dir``. creates the directory if not existent.
    returns the output directory as a ``Path``."""
    output_dir = Path(output_dir)
    if isinstance(input_dir_or_files, (str, Path)):
        input_dir = Path(input_dir_or_files)
        paths  = _images.collect_image_files(input_dir, suffixes=suffixes)
    else:
        paths = tuple(Path(path) for path in input_dir_or_files)

    images = _images.load_images(paths)
    image_names = _fileutils.unique_names_from_path(paths)

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
    min_valid_points: Optional[int] = None,
    separate_sides: Optional[bool] = None,
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
        min_valid_points=min_valid_points,
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
    file_type: ResultsFileType,
    resize: bool = True
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
        resize=resize
    ):
        if roiset.total_frames == 1:
            outbase = Path(roiset.image_name).stem
        else:
            digits = _fileutils.required_number_of_digits(roiset.total_frames)
            outbase = f"{Path(roiset.image_name).stem}_frame{str(roiset.frame_idx).zfill(digits)}"
        suffix = _fileutils.get_roi_file_suffix(file_type)
        outfile = output_dir / f"{outbase}_rois{suffix}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        roiset.to_file(outfile, file_type)
    return output_dir


def run_packaging_all_results(
    metadata_dir: PathLike,
    landmarks_dir: PathLike,
    alignment_dir: PathLike,
    rois_dir: PathLike,
    output_dir: PathLike,
    filetype: ResultsFileType
) -> Path:
    # TODO:
    #
    # may need to determine the strategy
    # of what image sizes to be used
    # (i.e. in 512 x 512, or the source frame size)
    #
    # the landmarks / alignment coordinates
    # may need to be converted accordingly...
    #

    metadata_dir = Path(metadata_dir)
    landmarks_dir = Path(landmarks_dir)
    alignment_dir = Path(alignment_dir)
    rois_dir = Path(rois_dir)
    output_dir = Path(output_dir)

    if filetype == 'hdf':
        _write_to_file = _packaging.package_hdf
    elif filetype == 'matlab':
        _write_to_file = _packaging.package_matfile
    else:
        raise ValueError(f"unexpected file type: '{filetype}'")

    # load: resized-images, metadata, landmarks, alignment
    metadata_table_path = _images.collected_images_metadata_path(metadata_dir)
    alignment_table_path = _landmarks.alignment_table_path(alignment_dir)
    dlcoutput = _landmarks.DLCOutput.from_directory(landmarks_dir)
    collected_images_path = _images.collected_images_video_path(metadata_dir)
    landmarks_video_path  = _landmarks.predicted_landmarks_video_path(landmarks_dir)
    alignment_video_path  = _landmarks.aligned_landmarks_video_path(alignment_dir)

    metadata  = _images.load_metadata_table(metadata_table_path)
    landmarks = _landmarks.landmarks_from_dlc_output(dlcoutput)
    alignment = _landmarks.load_alignment_table(alignment_table_path)
    source_images    = _iio.imread(str(collected_images_path))
    landmarks_images = _iio.imread(str(landmarks_video_path))
    alignment_images = _iio.imread(str(alignment_video_path))

    # for each source frame:
    # 1. find roi HDF5 file and read ROIs from it
    # 2. create the dict object containing:
    #    - landmarks (in 512 x 512),
    #    - reference-to-data alignment (in 512 x 512)
    #    - rois (in the size registered in the ROIs file)
    def _get_roifile(row: _pd.Series) -> Tuple[str, Path]:
        name = str(Path(row.Image).with_suffix(''))
        if row.TotalFrames == 1:
            roibase = name
        else:
            digits = _fileutils.required_number_of_digits(row.TotalFrames)
            roibase = f"{name}_frame{str(row.Frame).zfill(digits)}"
        return name, (rois_dir / f"{roibase}_rois.h5")

    for idx, row in metadata.iterrows():
        basename, roifile = _get_roifile(row)
        results = _packaging.Results(
            name=basename,
            images=_packaging.ResultImages(
                source=source_images[idx],
                landmarks=landmarks_images[idx],
                alignment=alignment_images[idx]
            ),
            landmarks=landmarks[idx],
            alignment=alignment[idx],
            rois=_rois.ROISet.load_hdf(roifile),
            datatype='512'  # NOTE: assumes 512x512 for the time being
        )
        _write_to_file(results, output_dir=output_dir)
