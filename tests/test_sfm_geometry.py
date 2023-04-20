import numpy as np
import cv2
import pytest

from icepy4d.sfm.geometry import *


def test_estimate_pose_minimum_kpts():
    kpts0 = np.array([[0, 0], [0, 1]])
    kpts1 = np.array([[0, 0], [0, 1]])
    K0 = np.eye(3)
    K1 = np.eye(3)
    thresh = 0.5
    conf = 0.9999

    result = estimate_pose(kpts0, kpts1, K0, K1, thresh, conf)
    assert result is None, "Expected None for less than 5 matched points"


def test_estimate_pose_valid_inputs():
    kpts0 = np.array(
        [[1853, 2632], [2122, 2744], [416, 2867], [1880, 2582], [2100, 2770]]
    ).astype(np.float32)

    kpts1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]])
    K0 = np.eye(3)
    K1 = np.eye(3)
    thresh = 0.5
    conf = 0.9999

    result = estimate_pose(kpts0, kpts1, K0, K1, thresh, conf)
    assert result is not None, "Unexpected None for valid inputs"


def test_project_points():
    # Create a dummy Camera object
    K = np.array(
        [
            [6.62649655e03, 0.00000000e00, 3.01324420e03],
            [0.00000000e00, 6.62649655e03, 1.94347461e03],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ).astype(np.float32)
    R = np.array(
        [
            [0.56442669, 0.82207137, -0.07497444],
            [-0.12850415, -0.0022157, -0.9917065],
            [-0.81541964, 0.56938015, 0.10438898],
        ]
    ).astype(np.float32)
    t = (
        np.array([[-160.33907216], [110.630176], [57.94062222]])
        .astype(np.float32)
        .reshape(3, 1)
    )
    camera = Camera(K=K, R=R, t=t, width=6012, height=4008, calib_path=None, dist=None)

    # Define some 3D points to be projected
    points3d = np.array(
        [
            [-15.5549, 315.317, 117.3462],
            [41.5596, 274.765, 92.0252],
            [53.2401, 163.044, 78.296],
            [-47.1266, 257.753, 119.0749],
            [-21.9544, 227.595, 95.207],
            [-19.5291, 238.100, 96.6108],
            [8.2768, 244.675, 80.6632],
            [-6.8296, 285.013, 98.518],
        ]
    ).astype(np.float32)

    # Project the 3D points to 2D image coordinates
    projected_points = project_points(points3d, camera)

    # Check if the shape of the projected points is correct
    assert projected_points.shape == (
        8,
        2,
    ), f"Unexpected shape: {projected_points.shape}"

    # Check if the projected points are within the image dimensions
    assert np.all(
        projected_points >= 0
    ), "Projected points are outside the image (negative values)"
    assert np.all(
        projected_points[:, 0] <= 6012
    ), "Projected points are outside the image (x-coordinate)"
    assert np.all(
        projected_points[:, 1] <= 4008
    ), "Projected points are outside the image (y-coordinate)"
