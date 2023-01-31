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

from typing import List

from .geometry import undistort_points
from .interpolate_colors import interpolate_point_colors
from ..classes.camera import Camera
from ..utils.utils import (
    convert_from_homogeneous,
    convert_to_homogeneous,
)
from ..thirdparty.triangulation import iterative_LS_triangulation

""" Triangulation class """


class Triangulate:
    def __init__(
        self,
        cameras: List[Camera] = None,
        image_points: List[np.ndarray] = None,
    ) -> None:
        """Inizialize class
        Parameters
        ----------
        cameras : List
            cameras is a list of Camera objects instances, each containing at least
            the projection matrix P
        image_points: List
            image_points is a list of nx2 np.arrays containing image coordinates of the features on the two images, as
            [ np.array([x1, y1]), np.array([x2, y2]) ],
        """
        self.cameras = cameras
        self.image_points = image_points
        self.points3d = None
        self.colors = None

    def triangulate_two_views(
        self,
        views_ids: List[int] = [0, 1],
        approach: str = "iterative_LS_triangulation",
        compute_colors: bool = False,
        image: np.ndarray = None,
        cam_id: int = 0,
    ) -> np.ndarray:

        if approach == "iterative_LS_triangulation":
            pts0_und = undistort_points(
                self.image_points[views_ids[0]], self.cameras[views_ids[0]]
            )
            pts1_und = undistort_points(
                self.image_points[views_ids[1]], self.cameras[views_ids[1]]
            )
            pts3d, ret = iterative_LS_triangulation(
                pts0_und,
                self.cameras[views_ids[0]].P,
                pts1_und,
                self.cameras[views_ids[1]].P,
            )
            logging.info(f"Point triangulation succeded: {ret.sum()/ret.size}.")

            self.points3d = pts3d
            if compute_colors:
                assert (
                    image is not None and type(image) == np.ndarray
                ), "Invalid input image for interpolating point colors"
                self.interpolate_colors_from_image(
                    image,
                    self.cameras[cam_id],
                )

            return self.points3d

        elif approach == "linear_triangulation":
            pts0_und = undistort_points(
                self.image_points[views_ids[0]], self.cameras[views_ids[0]]
            )
            pts1_und = undistort_points(
                self.image_points[views_ids[1]], self.cameras[views_ids[1]]
            )
            pts0_und = convert_to_homogeneous(pts0_und.T).T
            pts1_und = convert_to_homogeneous(pts1_und.T).T

            points3d = triangulate_points_linear(
                self.cameras[views_ids[0]].P,
                self.cameras[views_ids[1]].P,
                pts0_und,
                pts1_und,
            )
            self.points3d = convert_from_homogeneous(points3d.T).T

        return self.points3d

    def triangulate_nviews(self):
        """
        Triangulate a point visible in n camera views.
        P is a list of camera projection matrices.
        ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
        ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
        len of ip must be the same as le~n of P
        """
        P = [cam.P for cam in self.cameras]
        ip = self.image_points

        points3d = triangulate_nviews(P, ip)

        return points3d

    def interpolate_colors_from_image(
        self, image: np.ndarray, camera: Camera, convert_BRG2RGB: bool = True
    ):
        assert (
            self.points3d is not None
        ), "points 3D are not available, \
                Triangulate homologous points first."
        self.colors = interpolate_point_colors(
            self.points3d,
            image,
            camera,
            convert_BRG2RGB=convert_BRG2RGB,
        )
        logging.info(f"Point colors interpolated")

        return self.colors


""" Functions """


def triangulate_points_linear(P1, P2, x1, x2):
    """
    Two-view triangulation of points in
    x1,x2np.array([[274.128, 624.409]]) (nx3 homog. coordinates).
    Similar to openCV triangulatePoints.
    """
    if not len(x2) == len(x1):
        raise ValueError("Number of points don't match.")
    X = [triangulate_nviews([P1, P2], [x[0], x[1]]) for x in zip(x1, x2)]
    return np.array(X)


def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError("Number of points and number of cameras not equal.")
    n = len(P)
    M = np.zeros([3 * n, 4 + n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3 * i : 3 * i + 3, :4] = p
        M[3 * i : 3 * i + 3, 4 + i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]


if __name__ == "__main__":
    print("Test class")
