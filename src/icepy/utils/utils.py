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

import cv2
import numpy as np

from copy import deepcopy
from pathlib import Path


# --- File system ---#


def create_directory(path):
    """
    Creates a directory, if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- MAT ---#


def convert_to_homogeneous(x):
    """
    Convert 2xn or 3xn vector of n points in euclidean coordinates
    to a 3xn or 4xn vector homogeneous by adding a row of ones
    """
    x = np.array(x)
    ndim, npts = x.shape
    if ndim != 2 and ndim != 3:
        print(
            "Error: wrong number of dimension of the input vector.\
              A number of dimensions (rows) of 2 or 3 is required."
        )
        return None
    x1 = np.concatenate((x, np.ones((1, npts), "float32")), axis=0)
    return x1


def convert_from_homogeneous(x):
    """
    Convert 3xn or 4xn vector of n points in homogeneous coordinates
    to a 2xn or 3xn vector in euclidean coordinates, by dividing by the
    homogeneous part of the vector (last row) and removing one dimension
    """
    x = np.array(x)
    ndim, npts = x.shape
    if ndim != 3 and ndim != 4:
        print(
            "Error: wrong number of dimension of the input vector.\
              A number of dimensions (rows) of 2 or 3 is required."
        )
        return None
    x1 = x[: ndim - 1, :] / x[ndim - 1, :]
    return x1


def skew_symmetric(x):
    """Return skew symmetric matrix from input matrix x"""
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def compute_rmse(observed, predicted):
    """Compute RMSE between predicted and observed values"""
    return np.sqrt(((observed - predicted) ** 2).mean())


def compute_reprojection_error(observed, projected):
    """Compute reprojection error
    Parameters
    ----------
    observed : nx2 numpy array of float32
        array of observed image coordinates (usually, detected keypoints)
    projected : nx2 numpy array of float32
        array of image coordinates of projected points

    Returns
    -------
    err : nx3 numpy array of float32
        Reprojection error as in x, y direction and magnitude
    rmse : 2x1 numpy array of float32
      RMSE of the reprojection error in x, y directions
    """
    npts = len(observed)

    err = np.zeros((npts, 3), "float32")
    err[:, 0:2] = observed - projected
    err[:, 2:3] = np.linalg.norm(err[:, 0:2], axis=1).reshape((npts, 1))

    rmse = np.zeros((2, 1), "float32")
    for i in range(2):
        rmse[i] = compute_rmse(observed[:, i], projected[:, i])

    return err, rmse


# --- Homography warping based on estimated exterior orientation ---#


def homography_warping(
    cam_0: np.ndarray,
    cam_1: np.ndarray,
    image: np.ndarray,
    out_path: str = None,
) -> np.ndarray:

    print("Performing homography warping based on extrinsics matrix...")

    # Create deepcopies to not modify original data
    cam_0_ = deepcopy(cam_0)
    cam_1_ = deepcopy(cam_1)

    T = np.linalg.inv(cam_0_.pose)
    cam_0_.update_extrinsics(cam_0_.pose_to_extrinsics(T @ cam_0_.pose))
    cam_1_.update_extrinsics(cam_1_.pose_to_extrinsics(T @ cam_1_.pose))

    R = cam_1_.R
    K = cam_1_.K
    H = (cam_0_.K @ R) @ np.linalg.inv(K)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    w, h = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (h, w))
    if out_path is not None:
        cv2.imwrite(out_path, warped_image)
        print(f"Warped image {Path(out_path).stem} exported correctely")

    return cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)


# ---- Miscellaneous ---##


def PrintMatrix(mat: np.ndarray, num_decimals: int = 3) -> None:
    for row in mat:
        for el in row:
            print(f"{el:= 0.{num_decimals}f}\t", end=" ")
        print("")
