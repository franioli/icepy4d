"""
MIT License

Copyright (c) 2022 Francesco Ioli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import logging

from .camera import Camera


class Point:
    """
    Point Class for storing 3D point information and relative modules for getting image projections and building point clouds
    """

    def __init__(
        self, track_id: int, coord: np.ndarray, cov: np.ndarray = None
    ) -> None:
        self._track_id = track_id
        self._cood = coord
        if cov is not None:
            self._cov = cov

    # Setters

    # Getters
    @property
    def track_id(self):
        return self._track_id

    @property
    def coord(self):
        return self._cood

    def project(self, camera: Camera) -> np.ndarray:
        """
        project project the 3D point to the camera and return image coordinates

        Args:
            camera (Camera): Camera object containing extrinsics and intrinsics

        Returns:
            np.ndarray: coordinates of the projection in px
        """
        return camera.project_point(self._coord)
