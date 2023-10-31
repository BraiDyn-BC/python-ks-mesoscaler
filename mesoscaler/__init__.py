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

from . import (  # noqa: F401
    images,
    landmarks,
    atlas,
    procs,
    rois,
)

# image-related classes and procedures
InputImageFiles = images.InputImageFiles
InputImages     = images.InputImages
collect_image_files = images.collect_image_files
load_images = images.load_images

# landmark-related classes and procedures
DLCOutput = landmarks.DLCOutput
Landmarks = landmarks.Landmarks
Alignment = landmarks.Alignment
predict_dlc_landmarks = landmarks.predict_dlc_landmarks
align_dlc_landmarks   = landmarks.align_dlc_landmarks
update_dlc_landmarks  = landmarks.update_dlc_landmarks

# atlas_related classes and procedures
Atlas = atlas.Atlas

# roi-related classes and procedures
ROISet = rois.ROISet
generate_rois_batch  = rois.generate_rois_batch
generate_rois_single = rois.generate_rois_single

# procedures
run_image_collection    = procs.run_image_collection
run_landmark_prediction = procs.run_landmark_prediction
run_landmark_alignment  = procs.run_landmark_alignment
run_rois_generation     = procs.run_rois_generation
