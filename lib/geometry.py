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

from pathlib import Path
import numpy as np
import cv2
import pydegensac

from lib.classes import Camera
from lib.misc import compute_rmse

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.9999):
    """
    Estimate camera pose given matched points and intrinsics matrix.
    """
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None
    
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret
  

def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as le~n of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]


def triangulate_points_linear(P1, P2, x1, x2):
    """
    Two-view triangulation of points in
    x1,x2np.array([[274.128, 624.409]]) (nx3 homog. coordinates).
    Similar to openCV triangulatePoints.
    """
    if not len(x2) == len(x1):
        raise ValueError("Number of points don't match.")
    X = [triangulate_nviews([P1, P2], [x[0], x[1]]) for x in zip(x1, x2)]
    return np.array(X)
        

def project_points(points3d, camera: Camera):
    '''
    Project 3D points (Nx3 array) to image coordinates, given a Camera object
    Returns: 2D projected points (Nx2 array) in image coordinates
    '''
    rvec, _ = cv2.Rodrigues(camera.R)
    tvec = camera.t
    m, jacobian = cv2.projectPoints(np.expand_dims(points3d, 1), 
                                    rvec, tvec, 
                                    camera.K, camera.dist,
                                    )
    m = m[:,0,:]
    return m.astype('float32')


# def project_points(points3d, P, K=None, dist=None):
#     ''' Erroneous and substitued with new function
#     Project 3D points (Nx3 array) to image coordinates, given the projection matrix P (4x3 matrix)
#     If K matric and dist vector are given, the function computes undistorted image projections (otherwise, zero distortions are assumed)
#     Returns: 2D projected points (Nx2 array) in image coordinates
#     '''
#     points3d = cv2.convertPointsToHomogeneous(points3d)[:,0,:]
#     m = np.matmul(P, points3d.T)
#     m = m[0:2,:] / m[2,:]; 
#     m = m.astype(float).T
    
#     if dist is not None and K is not None:
#         m = cv2.undistortPoints(m, K, dist, None, K)[:,0,:]
        
#     return m.astype(float)

def undistort_points(pts, camera: Camera):
    ''' Wrapper around OpenCV cv2.undistortPoints to simplify function calling
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
    '''
    pts_und = cv2.undistortPoints(pts, camera.K, camera.dist, 
                                  None, camera.K)[:,0,:]                           
    return pts_und.astype('float32')


def undistort_image(image, camera: Camera, out_path=None):
    ''' Wrapper around OpenCV cv2.undistort function for simply undistorting an image
    Parameters
    ----------
    image : 2D numpy array
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

    '''
    image_und = cv2.undistort(image, camera.K, camera.dist, 
                              None, camera.K
                              )
    if out_path is not None:
        cv2.imwrite(out_path, image_und)
        
    return image_und

def undistort_image_new_cam_matrix(image, K, dist, downsample=1, out_path=None):
    ''' Deprecated, substituted with undistort_image()
    Undistort image with OpenCV
    '''
    h, w, _ = image.shape
    h_new, w_new = h*downsample, w*downsample
    K_scaled, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (int(w_new), int(h_new)))
    und = cv2.undistort(image, K, dist, None, K_scaled)
    x, y, w, h = roi
    und = und[y:y+h, x:x+w]  
    if out_path is not None:
        cv2.imwrite(out_path, und)
    return und, K_scaled
    # cam = 1
    # image = images[cam][0]
    # K, dist = cameras[cam][0].K, cameras[cam][0].dist
    # image_und = cv2.undistort(image, K, dist, None, K)
    # cv2.imwrite(images[cam].get_image_stem(0)+'_undistorted.tif', image_und)
    
def scale_intrinsics(K, scales):
    """ 
    Scale camera intrisics matrix (K) after image downsampling or upsampling.
    """
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)

def compute_reprojection_error(observed, projected):
    ''' Compute reprojection error
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
    '''
   
    npts = len(observed)
    
    err = np.zeros((npts,3), 'float32')
    err[:, 0:2] = observed - projected  
    err[:, 2:3] = np.linalg.norm(err[:,0:2] , axis=1).reshape((npts,1))
    
    rmse = np.zeros((2,1), 'float32')
    for i in range(2):
        rmse[i] = compute_rmse(observed[:,i], projected[:,i])
    
    return err, rmse
    