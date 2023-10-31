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

IMAGE_SUFFIXES = ('.png', '.tif', '.tiff')
VIDEO_FRAME_SIZE = (512, 512)
VIDEO_FRAME_RATE = 1

COLLECTED_IMAGES_VIDEO_NAME = "images.mp4"
COLLECTED_IMAGES_METADATA_NAME = "metadata.csv"

PREDICTED_LANDMARKS_VIDEO_NAME = "images_with_landmarks.mp4"
PREDICTED_LANDMARKS_TABLE_NAME = "landmarks.csv"
LANDMARK_LIKELIHOOD_THRESHOLD  = 0.9999
ALIGNMENT_TABLE_NAME           = "reference_to_images_transform.csv"
ALIGNED_LANDMARKS_VIDEO_NAME   = "images_with_aligned_landmarks.mp4"
ALIGNED_LANDMARKS_TABLE_NAME   = "aligned_landmarks.csv"
