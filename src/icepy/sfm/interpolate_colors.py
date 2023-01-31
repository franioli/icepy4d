import cv2
import numpy as np

from copy import deepcopy
from pathlib import Path
from scipy.interpolate import interp2d

from .geometry import project_points
from ..classes.camera import Camera

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
