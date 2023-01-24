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

import os
import cv2
import numpy as np

from copy import deepcopy
from pathlib import Path
from scipy.interpolate import interp2d

from ..base_classes.camera import Camera
from ..geometry import project_points
from ..visualization.visualization import imshow_cv
from ..utils.timer import AverageTimer

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


def homography_warping(
    cam_0: np.ndarray,
    cam_1: np.ndarray,
    image: np.ndarray,
    out_path: str = None,
    timer: AverageTimer = None,
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
    else:
        imshow_cv(warped_image, convert_RGB2BRG=False)

    timer.update("Homography warping")

    return cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)


# --- Tiles ---#
# @TODO: TO be moved in a separate class


def generateTiles(
    image,
    rowDivisor=2,
    colDivisor=2,
    overlap=200,
    viz=False,
    out_dir="tiles",
    writeTile2Disk=True,
):
    assert not (image is None), "Invalid image input"

    image = image.astype("float32")
    H = image.shape[0]
    W = image.shape[1]
    DY = round(H / rowDivisor / 10) * 10
    DX = round(W / colDivisor / 10) * 10
    dim = (rowDivisor, colDivisor)

    # TODO: implement checks on image dimension
    # Check image dimensions
    # if not W % colDivisor == 0:
    #     print('Number of columns non divisible by the ColDivisor. Removing last column.')
    #     image = image[:, 0:-1]
    # if not H % rowDivisor == 0:
    #     print('Number of rows non divisible by the RowDivisor. Removing last row')
    #     image = image[0:-1, :]

    tiles = []
    limits = []
    for col in range(0, colDivisor):
        for row in range(0, rowDivisor):
            tileIdx = np.ravel_multi_index((row, col), dim, order="F")
            limits.append(
                (
                    max(0, col * DX - overlap),
                    max(0, row * DY - overlap),
                    max(0, col * DX - overlap) + DX + overlap,
                    max(0, row * DY - overlap) + DY + overlap,
                )
            )
            # print(f'Tile {tileIdx}: xlim = ({ limits[tileIdx][0], limits[tileIdx][2]}), ylim = {limits[tileIdx][1], limits[tileIdx][3]}')
            tile = image[
                limits[tileIdx][1] : limits[tileIdx][3],
                limits[tileIdx][0] : limits[tileIdx][2],
            ]
            tiles.append(tile)
            if writeTile2Disk:
                isExist = os.path.exists(out_dir)
                if not isExist:
                    os.makedirs(out_dir)
                cv2.imwrite(
                    os.path.join(
                        out_dir,
                        "tile_"
                        + str(tileIdx)
                        + "_"
                        + str(limits[tileIdx][0])
                        + "_"
                        + str(limits[tileIdx][1])
                        + ".jpg",
                    ),
                    tile,
                )

    return tiles, limits


# --- Color interpolation ---#


def interpolate_point_colors(points3d, image, camera: Camera, convert_BRG2RGB=True):
    """Interpolate color of a 3D sparse point cloud, given an oriented image
    Parameters
    ----------
    points3d : float32 array
        nx3 array with 3d world points coordinates
    image : numpy array
        image on which to interpolate colors. It can be either a color image
        (3 channels, either RGB or BRG) or a grayscale image (1 channel)
    camera : Camera Object
        Camera object containing intrisics and extrinsics parameters
    covert_BRG2RGB : bool
        Flag for converting BRG channels to RGB. Set it to True, when using images in OpenCV format.
    Returns: float32 array
        Nx(num_channels) colour matrix, as float numbers (normalized in [0,1])
    -------
    """

    assert image.ndim == 3, "invalid input image. Image has not 3 channel"

    if convert_BRG2RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    num_pts = len(points3d)
    if len(image.shape) < 3:
        num_channels = 1
    else:
        num_channels = image.shape[2]
    projections = project_points(points3d, camera)
    image = image.astype(np.float32) / 255.0

    col = np.zeros((num_pts, num_channels))
    for ch in range(num_channels):
        col[:, ch] = bilinear_interpolate(
            image[:, :, ch],
            projections[:, 0],
            projections[:, 1],
        )
    return col


def bilinear_interpolate(im, x, y):
    """Perform bilinear interpolation given a 2D array (single channel image)
    and x, y arrays of unstructured query points
    Parameters
    ----------
    im : float32
        Single channel image.
    x : float32
        nx1 array of x coordinates of query points.
    y : float32
        nx1 array of y coordinates of query points.

    Returns: nx1 array of the interpolated color
    -------
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def interpolate_point_colors_interp2d(pointxyz, image, P, K=None, dist=None, winsz=1):
    """' Deprecated function (too slow). Use bilinear_interpolate instead
    Interpolate color of a 3D sparse point cloud, given an oriented image
      Inputs:
       - Nx3 matrix with 3d world points coordinates
       - image as np.array in RGB channels
           NB: if the image was impotred with OpenCV, it must be converted
           from BRG color space to RGB
               cv2.cvtColor(image_und, cv2.COLOR_BGR2RGB)
       - camera interior and exterior orientation matrixes: K, R, t
       - distortion vector according to OpenCV
    Output: Nx3 colour matrix, as float numbers (normalized in [0,1])
    """

    assert P is not None, "invalid projection matrix"
    assert image.ndim == 3, "invalid input image. Image has not 3 channel"

    if K is not None and dist is not None:
        image = cv2.undistort(image, K, dist, None, K)

    numPts = len(pointxyz)
    col = np.zeros((numPts, 3))
    h, w, _ = image.shape
    projections = project_points(pointxyz, P, K, dist)
    image = image.astype(np.float32) / 255.0

    for k, m in enumerate(projections):
        kint = np.round(m).astype(int)
        i = np.array([a for a in range(kint[1] - winsz, kint[1] + winsz + 1)])
        j = np.array([a for a in range(kint[0] - winsz, kint[0] + winsz + 1)])
        if i.min() < 0 or i.max() > h or j.min() < 0 or j.max() > w:
            continue

        ii, jj = np.meshgrid(i, j)
        ii, jj = ii.flatten(), jj.flatten()
        for rgb in range(0, 3):
            colPatch = image[i[0] : i[-1] + 1, j[0] : j[-1] + 1, rgb]
            fcol = interp2d(i, j, colPatch, kind="linear")
            col[k, rgb] = fcol(m[0], m[1])
    return col


# ---- Miscellaneous ---##


def PrintMatrix(mat: np.ndarray, num_decimals: int = 3) -> None:
    for row in mat:
        for el in row:
            print(f"{el:= 0.{num_decimals}f}\t", end=" ")
        print("")
