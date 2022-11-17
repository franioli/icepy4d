"""
MIT License

Copyright (c) 2022 Francesco Ioli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the 'Software'), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import cv2
from typing import List

from lib.classes import Features
from lib.classes import CameraNew as Camera

# from lib.sfm.triangulation import Triangulate

from lib.geometry import (
    estimate_pose,
    undistort_image,
    undistort_points,
    project_points,
)

""" Single_camera_geometry class"""


class Single_camera_geometry:
    def __init__(
        self,
        camera: Camera,
    ) -> None:
        self.camera = camera

    def space_resection(
        self,
        image_points: np.ndarray,
        object_poits: np.ndarray,
        reprojection_error: float = 3.0,
    ) -> None:
        ret, r, t, inliers = cv2.solvePnPRansac(
            object_poits,
            image_points,
            self.camera.K,
            self.camera.dist,
            reprojectionError=reprojection_error,
        )
        if ret:
            print(
                f"Space resection succeded. Number of inlier points: {len(inliers)}/{len(object_poits)}"
            )
        else:
            print(
                "Space resection failed. Wrong input data or not enough inliers found"
            )
            return

        R, _ = cv2.Rodrigues(r)
        extrinsics = np.concatenate((R, t), axis=1)
        self.camera.build_camera_EO(extrinsics=extrinsics)


""" Two_view_geometry class"""


class Two_view_geometry:
    def __init__(self, cameras: List[Camera], features: List[np.ndarray]) -> None:
        """Inizialize class
        Parameters
        ----------
        cameras : List[Cameras]
            cameras is a list of Camera objects instances, each containing cameras
            intrisics and extrinsics orientation
        features: List[Features]
            features is a list containing the features (nx2 numpy array) matched on the two cameras
        """
        self.cameras = cameras
        self.features = features

    def relative_orientation(
        self, threshold: float = 1.0, confidence: float = 0.9999, scale_factor=None
    ) -> list:
        """Perform relative orientation with OpenCV recoverPose function
        Parameters
        ----------
        threshold : float (default = 1)
            Distance trheshold from epipolar line (in pixel) for reject outliers
            with RANSAC, when estimating camera poses
            See OpenCV recoverPose function for more information.
        confidence:  float (default = 0.9999)
            Confidence for RANSAC estimation
        scale_factor : float (default=None)
            Scale factor for scaling the two-view-geometry model
        Returns
        -------
        cameras :
        valid :

        """

        # Check if extrinsics matrix of camera 0 is available
        if self.cameras[0].extrinsics is None:
            print(
                "Extrinsics matrix is not available for camera 0. Please, compute it before running Two_view_geometry estimation."
            )
            return

        # Estimate Realtive Pose with Essential Matrix
        # R, t make up a tuple that performs a change of basis from the first camera's coordinate system to the second camera's coordinate system.
        R, t, valid = estimate_pose(
            self.features[0],
            self.features[1],
            self.cameras[0].K,
            self.cameras[1].K,
            thresh=threshold,
            conf=confidence,
        )
        print(f"Relative Orientation - valid points: {valid.sum()}/{len(valid)}")

        # If the scaling factor is given, scale the stereo model
        if scale_factor is not None:
            t = t * scale_factor
        else:
            print(
                "No scaling factor (e.g., computed from camera baseline) is provided. Two-view-geometry estimated up to a scale factor."
            )

        # Update Camera 1 Extrinsics and Pose relatevely to the world reference system (by multipling the estimated Pose with the Pose of Camera 0)
        # self.cameras[1].R = R
        # self.cameras[1].t = t.reshape(3, 1)
        # self.cameras[1].Rt_to_extrinsics()
        # self.cameras[1].extrinsics_to_pose()
        # cam2toWorld = self.cameras[0].pose @ self.cameras[1].pose
        # self.cameras[1].build_camera_EO(pose=cam2toWorld)

        # With New Camera class
        extrinsics = self.cameras[1].Rt_to_extrinsics(R, t)
        self.cameras[1].update_extrinsics(extrinsics)
        cam2toWorld = self.cameras[0].pose @ self.cameras[1].pose

        extrinsics = self.cameras[1].pose_to_extrinsics(cam2toWorld)
        self.cameras[1].update_extrinsics(extrinsics)
        return self.cameras, valid

    def scale_model_with_baseline(self, baseline_world):
        """Scale model given the camera baseline in thw world reference system
        Parameters
        ----------
        baseline_world : float
            Camera baseline in the world reference system
        Returns
        -------
        cameras :
        scale_fct :
        """

        baseline = np.linalg.norm(
            self.cameras[0].get_C_from_pose() - self.cameras[1].get_C_from_pose()
        )
        scale_fct = baseline_world / baseline

        T = np.eye(4)
        T[0:3, 0:3] = T[0:3, 0:3] * scale_fct

        # Apply scale to camera extrinsics and update camera proprierties
        self.cameras[1].pose[:, 3:4] = np.dot(T, self.cameras[1].pose[:, 3:4])
        self.cameras[1].pose_to_extrinsics()
        self.cameras[1].update_camera_from_extrinsics()

        return self.cameras, scale_fct
