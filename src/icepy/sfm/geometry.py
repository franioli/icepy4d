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
import cv2

from ..classes.camera import Camera


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.9999):
    """
    Estimate camera pose given matched points and intrinsics matrix.

    Args:
        kpts0 (np.ndarray): A Nx2 array of keypoints in the first image.
        kpts1 (np.ndarray): A Nx2 array of keypoints in the second image.
        K0 (np.ndarray): A 3x3 intrinsics matrix of the first camera.
        K1 (np.ndarray): A 3x3 intrinsics matrix of the second camera.
        thresh (float): The inlier threshold for RANSAC.
        conf (float, optional): The confidence level for RANSAC. Defaults to 0.9999.

    Returns:
        tuple: A tuple containing the rotation matrix, translation vector, and
        boolean mask indicating inliers.

        - R (np.ndarray): A 3x3 rotation matrix.
        - t (np.ndarray): A 3x1 translation vector.
        - inliers (np.ndarray): A boolean array indicating which keypoints are inliers.

    NOTE:
        R, t make up a tuple that performs a change of basis from the first camera's coordinate system to the second camera's coordinate system. Therefore, if the first camera has its own exterior orientation with respect to a world reference system, R and t estimated can be as the components of the extrinsics matrix (transformation from world to camera) of the second camera (be careful, R,t are NOT the components of the pose matrix of the second camera, because they describe transformation from 'first camera's coordinate system to the second'!). See https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0 for more info.
    """
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf, method=cv2.RANSAC
    )

    assert E is not None, "Unable to estimate Essential matrix"

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def project_points(points3d, camera: Camera):
    """
    Project 3D points (Nx3 array) to image coordinates, given a Camera object.

    Args:
        points3d (np.ndarray): A Nx3 array of 3D points.
        camera (Camera): A Camera object containing the camera's intrinsic and extrinsic parameters.

    Returns:
        np.ndarray: A Nx2 array of 2D projected points in image coordinates.
    """
    rvec, _ = cv2.Rodrigues(camera.R)
    tvec = camera.t
    m, jacobian = cv2.projectPoints(
        np.expand_dims(points3d, 1),
        rvec,
        tvec,
        camera.K,
        camera.dist,
    )
    m = m[:, 0, :]
    return m.astype("float32")


def undistort_points(pts, camera: Camera):
    """Wrapper around OpenCV cv2.undistortPoints to simplify function calling
    Parameters
    ----------
    pts : nx2 array of float32
        Array of distorted image points.
    camera : Camera object
        Camera object containing K and dist arrays.

    Returns
    -------
    pts : nx2 array of float32
        Array of undistorted image points.
    """
    pts_und = cv2.undistortPoints(pts, camera.K, camera.dist, None, camera.K)[:, 0, :]
    return pts_und.astype("float32")


def undistort_image(image, camera: Camera, out_path: str = None):
    """Wrapper around OpenCV cv2.undistort function for simply undistorting an image
    Parameters
    ----------
    image : 2D numpy array with BRG color channels (as default in OpenCV)
        Image.
    camera : Camera object
        Camera object containing K and dist arrays.
    out_path : Path or str, optional
        Path for writing the undistorted image to disk.
        The default is None (image is not written to disk)

    Returns
    -------
    image_und : 2D numpy array
        Undistorted image.

    """
    image_und = cv2.undistort(image, camera.K, camera.dist, None, camera.K)
    if out_path is not None:
        cv2.imwrite(out_path, image_und)

    return image_und


def undistort_image_new_cam_matrix(image, K, dist, downsample=1, out_path=None):
    """Deprecated, substituted with undistort_image()
    Undistort image with OpenCV
    """
    h, w, _ = image.shape
    h_new, w_new = h * downsample, w * downsample
    K_scaled, roi = cv2.getOptimalNewCameraMatrix(
        K, dist, (w, h), 1, (int(w_new), int(h_new))
    )
    und = cv2.undistort(image, K, dist, None, K_scaled)
    x, y, w, h = roi
    und = und[y : y + h, x : x + w]
    if out_path is not None:
        cv2.imwrite(out_path, und)
    return und, K_scaled
    # cam = 1
    # image = images[cam].read_image(epoch).value
    # K, dist = cameras[cam][0].K, cameras[cam][0].dist
    # image_und = cv2.undistort(image, K, dist, None, K)
    # cv2.imwrite(images[cam].get_image_stem(0)+'_undistorted.tif', image_und)


def scale_intrinsics(K, scales):
    """
    Scale camera intrisics matrix (K) after image downsampling or upsampling.
    """
    scales = np.diag([1.0 / scales[0], 1.0 / scales[1], 1.0])
    return np.dot(scales, K)
