'''
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
'''

import numpy as np

from lib.classes import Camera
# from lib.sfm.triangulation import Triangulate

from lib.geometry import (estimate_pose,
                          undistort_image,
                          undistort_points,
                          project_points,
                          )

''' Two_view_geometry class'''


class Two_view_geometry():
    def __init__(self, cameras: list, features: list) -> None:
        ''' Inizialize class
        Parameters
        ----------
        cameras : List
            cameras is a list of Camera objects instances, each containing cameras 
            intrisics and extrinsics orientation
        features: List
            features is a list of Features object detected (and matched) for 
            every the cameras
        '''
        self.cameras = cameras
        self.features = features

    def relative_orientation(self, threshold=1., confidence=0.9999) -> list:
        ''' Perform relative orientation with OpenCV recoverPose function
        Parameters
        ----------
        threshold : float (default = 1)
            Distance trheshold from epipolar line (in pixel) for reject outliers
            with RANSAC, when estimating camera poses
            See OpenCV recoverPose function for more information.
        confidence:  float (default = 0.9999)
            Confidence for RANSAC estimation

        Returns
        -------
        cameras : 
        valid : 

        '''
        # Estimate Realtive Pose with Essential Matrix
        R, t, valid = estimate_pose(
            self.features[0],
            self.features[1],
            self.cameras[0].K,
            self.cameras[1].K,
            thresh=threshold, conf=confidence,
        )
        print(
            f'Relative Orientation - valid points: {valid.sum()}/{len(valid)}'
        )

        # Update camera 1 extrinsics
        self.cameras[1].R = R
        self.cameras[1].t = t.reshape(3, 1)
        self.cameras[1].Rt_to_extrinsics()
        self.cameras[1].extrinsics_to_pose()

        return self.cameras, valid

    def absolute_orientation(self) -> None:
        pass

    def scale_model_with_baseline(self, baseline_world):
        ''' Scale model given the camera baseline in thw world reference system
        Parameters
        ----------
        baseline_world : float 
            Camera baseline in the world reference system
        Returns
        -------
        cameras : 
        scale_fct :           
        '''

        baseline = np.linalg.norm(
            self.cameras[0].get_C_from_pose() -
            self.cameras[1].get_C_from_pose()
        )
        scale_fct = baseline_world / baseline

        T = np.eye(4)
        T[0:3, 0:3] = T[0:3, 0:3] * scale_fct

        # Apply scale to camera extrinsics and update camera proprierties
        self.cameras[1].pose[:, 3:4] = np.dot(
            T, self.cameras[1].pose[:, 3:4]
        )
        self.cameras[1].pose_to_extrinsics()
        self.cameras[1].update_camera_from_extrinsics()

        return self.cameras, scale_fct
