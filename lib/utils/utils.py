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
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import time
import logging

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from PIL import Image
from rasterio.transform import Affine
from scipy.interpolate import (
    interp2d,
    LinearNDInterpolator,
)

from lib.base_classes.camera import Camera
from lib.geometry import project_points
from lib.visualization import imshow_cv


# ---- Timer and logging---##


class AverageTimer:
    """Class to help manage printing simple timing of code execution."""

    def __init__(self, smoothing=0.3):
        self.smoothing = smoothing
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name="default"):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text="Timer"):
        total = 0.0
        msg = f"[Timer] | [{text}] "
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                msg = msg + f"%s=%.3f, " % (key, val)
                total += val
        logging.info(msg)

        self.reset()


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


# ---- DSM and orthophotos ---##
class DSM:
    # @TODO: define new better class
    """Class to store and manage DSM."""

    def __init__(self, xx, yy, zz, res):
        # xx, yy = np.meshgrid(x,y)
        self.x = xx
        self.y = yy
        self.z = zz
        self.res = res


def build_dsm(
    points3d,
    dsm_step=1,
    xlim=None,
    ylim=None,
    interp_method="linear",
    fill_value=np.nan,
    save_path=None,
    make_dsm_plot=False,
):
    # TODO: Use Numpy binning instead of pandas grouping.

    def round_to_val(a, round_val):
        return np.round(np.array(a, dtype="float32") / round_val) * round_val

    # Check dimensions of input array
    assert np.any(np.array(points3d.shape) == 3), "Invalid size of input points"
    if points3d.shape[0] == points3d.shape[1]:
        print(
            "Warning: input vector has just 3 points. Unable to check validity of point dimensions."
        )
    if points3d.shape[0] == 3:
        points3d = points3d.T

    if save_path is not None:
        save_path = Path(save_path)
        save_fld = save_path.parent
        save_stem = save_path.stem

    # retrieve points and limits
    x, y, z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    if xlim is None:
        xlim = [np.floor(x.min()), np.ceil(x.max())]
    if ylim is None:
        ylim = [np.floor(y.min()), np.ceil(y.max())]

    n_pts = len(x)
    d_round = np.empty([n_pts, 3])
    d_round[:, 0] = round_to_val(points3d[:, 0], dsm_step)
    d_round[:, 1] = round_to_val(points3d[:, 1], dsm_step)
    d_round[:, 2] = points3d[:, 2]

    # sorting data
    ind = np.lexsort((d_round[:, 1], d_round[:, 2]))
    d_sort = d_round[ind]

    # making dataframes and grouping stuff
    df_cols = ["x_round", "y_round", "z"]
    df = pd.DataFrame(d_sort, columns=df_cols)
    group_xy = df.groupby(["x_round", "y_round"])
    group_mean = group_xy.mean()
    binned_df = group_mean.index.to_frame()
    binned_df["z"] = group_mean

    # Move again to numpy array for interpolation
    binned_arr = binned_df.to_numpy()
    x = binned_arr[:, 0].astype("float32")
    y = binned_arr[:, 1].astype("float32")
    z = binned_arr[:, 2].astype("float32")

    # Interpolate dsm
    xq = np.arange(xlim[0], xlim[1], dsm_step)
    yq = np.arange(ylim[0], ylim[1], dsm_step)
    grid_x, grid_y = np.meshgrid(xq, yq)

    if fill_value == "mean":
        fill_value = z.mean()

    interp = LinearNDInterpolator(
        list(zip(x, y)),
        z,
        fill_value=fill_value,
    )
    dsm_grid = interp(grid_x, grid_y)

    # plot dsm
    if make_dsm_plot:
        fig, ax = plt.subplots()
        dsm_plt = ax.contourf(grid_x, grid_y, dsm_grid)
        scatter = ax.scatter(
            points3d[:, 0],
            points3d[:, 1],
            s=5,
            c=points3d[:, 2],
            marker="o",
            cmap="viridis",
            alpha=0.4,
            edgecolors="k",
        )
        ax.axis("equal")
        ax.invert_yaxis()
        # fig.colorbar(dsm_plt, cax=ax, orientation='vertical')
        cbar = plt.colorbar(dsm_plt, ax=ax)
        cbar.set_label("z")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("DSM interpolated from point cloud on plane X-Y")
        fig.tight_layout()
        # plt.show()
        if save_path is not None:
            plt.savefig(save_fld.joinpath(save_stem + "_plot.png"), bbox_inches="tight")

    # Save dsm as GeoTIff
    if save_path is not None:
        save_path = Path(save_path)
        create_directory(save_path.parent)
        rater_origin = [xlim[0] - dsm_step / 2, ylim[0] - dsm_step / 2]
        transform = Affine.translation(rater_origin[0], rater_origin[1]) * Affine.scale(
            dsm_step, -dsm_step
        )
        mask = np.invert(np.isnan(dsm_grid))
        rater_origin = [grid_x[0, 0], grid_y[0, 0]]
        transform = Affine.translation(rater_origin[0], rater_origin[1]) * Affine.scale(
            dsm_step, -dsm_step
        )
        with rasterio.open(
            save_path,
            "w",
            driver="GTiff",
            height=dsm_grid.shape[0],
            width=dsm_grid.shape[1],
            count=1,
            dtype="float32",
            # crs="EPSG:32632",
            transform=transform,
        ) as dst:
            dst.write(dsm_grid, 1)
            if fill_value is not None:
                mask = np.invert(np.isnan(dsm_grid))
                dst.write_mask(mask)

        if fill_value is not None:
            # mask = np.invert(np.isnan(dsm_grid))
            with rasterio.open(
                save_path.parent / (save_path.stem + "_msk.tif"),
                "w",
                driver="GTiff",
                height=dsm_grid.shape[0],
                width=dsm_grid.shape[1],
                count=1,
                dtype="float32",
                # crs="EPSG:32632",
                transform=transform,
            ) as dst:
                dst.write(mask, 1)

    # Return a DSM object
    dsm = DSM(grid_x, grid_y, dsm_grid, dsm_step)

    return dsm


def generate_ortophoto(
    image, dsm, camera: Camera, xlim=None, ylim=None, res=None, save_path=None
):
    xx = dsm.x
    yy = dsm.y
    zz = dsm.z
    if res is None:
        res = dsm.res

    if xlim is None:
        xlim = [xx[0, 0], xx[0, -1]]
    if ylim is None:
        ylim = [yy[0, 0], yy[-1, 0]]

    dsm_shape = dsm.x.shape
    ncell = dsm_shape[0] * dsm_shape[1]
    xyz = np.zeros((ncell, 3))
    xyz[:, 0] = xx.flatten()
    xyz[:, 1] = yy.flatten()
    xyz[:, 2] = zz.flatten()
    valid_cell = np.invert(np.isnan(xyz[:, 2]))

    cols = np.full((ncell, 3), 0.0, "float32")
    cols[valid_cell, :] = interpolate_point_colors(
        xyz[valid_cell, :],
        image,
        camera,
    )
    ortophoto = np.zeros((dsm_shape[0], dsm_shape[1], 3))
    ortophoto[:, :, 0] = cols[:, 0].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto[:, :, 1] = cols[:, 1].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto[:, :, 2] = cols[:, 2].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto = np.uint8(ortophoto * 255)

    # Save dsm as GeoTIff
    if save_path is not None:
        save_path = Path(save_path)
        create_directory(save_path.parent)
        rater_origin = [xlim[0] - res / 2, ylim[0] - res / 2]
        transform = Affine.translation(rater_origin[0], rater_origin[1]) * Affine.scale(
            res, -res
        )
        with rasterio.open(
            save_path,
            "w",
            driver="GTiff",
            height=ortophoto.shape[0],
            width=ortophoto.shape[1],
            count=3,
            dtype="uint8",
            # crs="EPSG:32632",
            transform=transform,
        ) as dst:
            dst.write(np.moveaxis(ortophoto, -1, 0))

    return ortophoto


# ---- Miscellaneous ---##


def PrintMatrix(mat: np.ndarray, num_decimals: int = 3) -> None:
    for row in mat:
        for el in row:
            print(f"{el:= 0.{num_decimals}f}\t", end=" ")
        print("")
