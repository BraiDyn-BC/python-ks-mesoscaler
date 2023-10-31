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
    base,
    paths,
    reference,
    prediction,
    alignment,
)

DLCOutput = base.DLCOutput
Landmarks = base.Landmarks
Alignment = base.Alignment

predict_dlc_landmarks = prediction.predict_dlc_landmarks
load_reference_landmarks  = base.load_reference_landmarks
landmarks_from_dlc_output = base.landmarks_from_dlc_output
write_labeled_video   = base.write_labeled_video
write_dlc_landmarks   = base.write_dlc_landmarks
write_alignment_table = base.write_alignment_table
load_alignment_table  = base.load_alignment_table
align_dlc_landmarks   = alignment.align_dlc_landmarks
update_dlc_landmarks  = alignment.update_dlc_landmarks

predicted_landmarks_video_path = paths.predicted_landmarks_video_path
predicted_landmarks_table_path = paths.predicted_landmarks_table_path
alignment_table_path           = paths.alignment_table_path
aligned_landmarks_video_path   = paths.aligned_landmarks_video_path
aligned_landmarks_table_path   = paths.aligned_landmarks_table_path
