import numpy as np

from typing import List
from lmfit import Parameters

from ..thirdparty.transformations import (
    euler_matrix,
)
from .utils import convert_to_homogeneous


def compute_tform_matrix_from_params(params: Parameters) -> np.ndarray:

    parvals = params.valuesdict()
    rx = parvals["rx"]
    ry = parvals["ry"]
    rz = parvals["rz"]
    tx = parvals["tx"]
    ty = parvals["ty"]
    tz = parvals["tz"]
    m = parvals["m"]

    # Build 4x4 transformation matrix (T) in homogeneous coordinates
    T = np.identity(4)
    R = euler_matrix(rx, ry, rz)
    T[0:3, 0:3] = ((1 + m) * np.identity(3)) @ R[:3, :3]
    T[0:3, 3:4] = np.array([tx, ty, tz]).reshape(3, 1)

    return T


def compute_residuals(
    params: Parameters,
    x0: np.ndarray,
    x1: np.ndarray,
    weights: np.ndarray = None,
    prior_covariance_scale: float = None,
) -> np.ndarray:
    """3D rototranslation with scale factor

    X1_ = T_ + m * R * X0_

    Inputs:
    - x0 (np.ndarray): Points in the starting reference system
    - x1 (np.ndarray): Points in final reference system
    - weights (np.ndarray, defult = None): weights (e.g., inverse of a-priori observation uncertainty)
    - prior_covariance_scale (float, default = None): A-priori sigma_0^2

    Return:
    - res (nx1 np.ndarray): Vector of the weighted residuals

    """

    # # Get parameters
    # parvals = params.valuesdict()
    # rx = parvals["rx"]
    # ry = parvals["ry"]
    # rz = parvals["rz"]
    # tx = parvals["tx"]
    # ty = parvals["ty"]
    # tz = parvals["tz"]
    # m = parvals["m"]

    # # Build 4x4 transformation matrix (T) in homogeneous coordinates
    # T = np.identity(4)
    # R = euler_matrix(rx, ry, rz)
    # T[0:3, 0:3] = (m * np.identity(3)) @ R[:3, :3]
    # T[0:3, 3:4] = np.array([tx, ty, tz]).reshape(3, 1)

    # Build 4x4 transformation matrix (T) in homogeneous coordinates
    T = compute_tform_matrix_from_params(params)

    # Convert points to homogeneos coordinates and traspose np array to obtain a 4xn matrix
    x0 = convert_to_homogeneous(x0).T

    # Apply transformation to x0 points
    x1_ = T @ x0
    x1_ = x1_[:3, :].T

    # Compute residuals as differences between observed and estimated values, scaled by the a-priori observation uncertainties
    res = x1 - x1_

    # If weigthts are provided, scale residual
    if weights is not None:

        if weights.shape != res.shape:
            raise ValueError(
                f"Wrong dimensions of the weight matrix. It must be of size {res.shape}"
            )

        res = res * weights

    return res.flatten()


def apply_transformation_to_points(
    points3d: np.ndarray,
    tform: np.ndarray,
) -> np.ndarray:

    points3d = convert_to_homogeneous(points3d)
    points_out = tform @ points3d.T
    print(points_out.shape)
    return points_out[0:3].T
