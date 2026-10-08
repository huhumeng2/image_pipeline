# Copyright (c) 2026, Yong Ling
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following disclaimer
#     in the documentation and/or other materials provided with the distribution.
#   * Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Test camera image archive export with the current NumPy bytes API."""

from io import BytesIO
import tarfile
import unittest
from unittest.mock import patch

from camera_calibration.mono_calibrator import MonoCalibrator
from camera_calibration.stereo_calibrator import StereoCalibrator
import cv2
import numpy as np


class EncodedImage:
    """Model NumPy's encoded image API after removal of tostring."""

    def __init__(self, image):
        """Wrap an encoded OpenCV image."""
        self.image = image

    def tobytes(self):
        """Return the encoded bytes."""
        return self.image.tobytes()


class TestCalibrationArchiveBytes(unittest.TestCase):
    """Verify mono and stereo image payloads in saved calibration archives."""

    def test_saved_mono_and_stereo_images_roundtrip(self):
        """Decode each saved PNG and compare it with the source image."""
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        encoded = cv2.imencode('.png', image)[1]
        for cls in (MonoCalibrator, StereoCalibrator):
            with self.subTest(calibrator=cls.__name__):
                calibrator = cls.__new__(cls)
                calibrator.db = (
                    [(None, image)]
                    if cls is MonoCalibrator
                    else [(None, image, image)]
                )
                calibrator.yaml = lambda *args: 'camera_name: test\n'
                calibrator.ost = lambda: 'calibration\n'
                if cls is StereoCalibrator:
                    setattr(calibrator, 'l', object())
                    calibrator.r = object()
                buffer = BytesIO()
                with patch(
                    'cv2.imencode', return_value=(True, EncodedImage(encoded))
                ):
                    with tarfile.open(fileobj=buffer, mode='w') as archive:
                        calibrator.do_tarfile_save(archive)
                buffer.seek(0)
                with tarfile.open(fileobj=buffer, mode='r') as archive:
                    names = [
                        name
                        for name in archive.getnames()
                        if name.endswith('.png')
                    ]
                    self.assertEqual(
                        len(names), 1 if cls is MonoCalibrator else 2
                    )
                    for name in names:
                        payload = archive.extractfile(name).read()
                        decoded = cv2.imdecode(
                            np.frombuffer(payload, np.uint8),
                            cv2.IMREAD_GRAYSCALE,
                        )
                        np.testing.assert_array_equal(decoded, image)
