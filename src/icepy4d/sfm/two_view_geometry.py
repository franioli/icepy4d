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

import importlib
import logging
from typing import List

import cv2
import numpy as np

from ..classes.camera import Camera
from .geometry import estimate_pose

""" RelativeOrientation class"""


class RelativeOrientation:
    def __init__(self, cameras: List[Camera], features: List[np.ndarray]) -> None:
        """Inizialize RelativeOrientation class
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

    def estimate_pose(
        self,
        threshold: float = 1.0,
        confidence: float = 0.9999,
        scale_factor=None,
    ) -> list:
        """Perform relative orientation with OpenCV recoverPose function
        Parameters
        ----------
        threshold : float (default = 1)
            Maximum distance from epipolar line (in pixel) for rejecting outliers with RANSAC, when estimating camera poses
            See OpenCV recoverPose function for more information.
        confidence:  float (default = 0.9999)
            Confidence for RANSAC estimation
        scale_factor : float (default=None)
            Scale factor for scaling the two-view-geometry model
        Returns
        -------
            valid : valid features used for estimating relative orientation

        """

        # Check if extrinsics matrix of camera 0 is available
        assert self.cameras[0].extrinsics is not None, print(
            "Extrinsics matrix is not available for camera 0. Please, compute it before running RelativeOrientation estimation."
        )

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
        logging.info(f"Relative Orientation - valid points: {valid.sum()}/{len(valid)}")

        # If the scaling factor is given, scale the stereo model
        if scale_factor is not None:
            t = t * scale_factor
        else:
            logging.warning(
                "No scaling factor (e.g., computed from camera baseline) is provided. Two-view-geometry estimated up to a scale factor."
            )

        # Update Camera 1 Extrinsics and Pose relatevely to the world reference system (by multipling the estimated Pose with the Pose of Camera 0)
        extrinsics = self.cameras[1].Rt_to_extrinsics(R, t)
        self.cameras[1].update_extrinsics(extrinsics)
        cam2toWorld = self.cameras[0].pose @ self.cameras[1].pose

        extrinsics = self.cameras[1].pose_to_extrinsics(cam2toWorld)
        self.cameras[1].update_extrinsics(extrinsics)

        logging.info("Relative orientation Succeded.")

        return valid

    def get_scale_factor_from_baseline(self, baseline_world: float):
        """Scale model given the camera baseline in thw world reference system
        Parameters
        ----------
        baseline_world : float
            Camera baseline in the world reference system
        Returns
        -------
            scale_fct : float
        """

        baseline = np.linalg.norm(self.cameras[0].C - self.cameras[1].C)
        scale_fct = baseline_world / baseline

        return scale_fct

    def estimate_F_matrix(
        self,
        threshold: float = 1,
        confidence: float = 0.9999,
        max_iters: int = 10000,
        laf_consistensy_coef: float = -1.0,
        error_type: str = "sampson",
        symmetric_error_check: bool = True,
        enable_degeneracy_check: bool = True,
    ) -> List[np.ndarray]:
        """
        Computes the fundamental matrix and inliers between the two images using geometric verification.

        Args:
            threshold (float): Pixel error threshold for considering a correspondence an inlier.
            confidence (float): The required confidence level in the results.
            max_iters (int): The maximum number of iterations for estimating the fundamental matrix.
            laf_consistensy_coef (float): The weight given to Local Affine Frame (LAF) consistency term for pydegensac.
            error_type (str): The error function used for computing the residuals in the RANSAC loop.
            symmetric_error_check (bool): If True, performs an additional check on the residuals in the opposite direction.
            enable_degeneracy_check (bool): If True, enables the check for degeneracy using SVD.

        Returns:
            np.ndarray: A Boolean array that masks the correspondences that were identified as inliers.

        TODO: This method should be just a wrapper around the geometric_verification. Use a separate geometric_verification function instead.
        """

        try:
            pydegensac = importlib.import_module("pydegensac")
            use_pydegensac = True
        except:
            logging.error(
                "Pydegensac not available. Using MAGSAC++ (OpenCV) for geometric verification."
            )
            use_pydegensac = False
        try:
            if use_pydegensac:
                self.F, self.inlMask = pydegensac.findFundamentalMatrix(
                    self.features[0],
                    self.features[1],
                    px_th=threshold,
                    conf=confidence,
                    max_iters=max_iters,
                    laf_consistensy_coef=laf_consistensy_coef,
                    error_type=error_type,
                    symmetric_error_check=symmetric_error_check,
                    enable_degeneracy_check=enable_degeneracy_check,
                )
                logging.info(
                    f"Pydegensac found {self.inlMask.sum()} inliers ({self.inlMask.sum()*100/len(self.features[0]):.2f}%)"
                )
            else:
                self.F, inliers = cv2.findFundamentalMat(
                    self.features[0],
                    self.features[1],
                    cv2.USAC_MAGSAC,
                    0.5,
                    0.999,
                    100000,
                )
                self.inlMask = inliers > 0
                logging.info(
                    f"MAGSAC++ found {self.inlMask.sum()} inliers ({self.inlMask.sum()*100/len(self.features[0]):.2f}%)"
                )
            self.features[0] = self.features[0][self.inlMask]
            self.features[1] = self.features[1][self.inlMask]
        except ValueError as err:
            logging.error(f"Unable to perform geometric verification: {err}.")

        return (self.F, self.inlMask)
