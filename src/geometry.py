from pathlib import Path
import numpy as np
import cv2


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
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

def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)

def P_from_KRT(K, R, t):
    """
    Return the 4x3 P matrix from 3x3 K matrix, 3x3 R matrix and 3x1 t vector, as:
        K[ R | t ]
    """
    RT = np.zeros((3,4))
    RT[:, 0:3] = R
    RT[:, 3:4] = t
    P = np.dot(K,RT) 
    return P

def make_P_homogeneous(P):
    """
    Return the 4x4 P matrix from 3x4 P matrix, as:
        [      P     ]
        [------------]
        [ 0  0  0  1 ]
    """
    P_hom = np.eye(4)
    P_hom[0:3, 0:3] = P
    return P_hom

def X0_from_P(P):
    """
    Compute camera perspective centre from 3x4 P matrix, as:
        X0 = - inv(KR) * Kt 
    """
    X0 = -np.matmul(np.linalg.inv(P[:,0:3]),P[:,3]).reshape(3,1)
    return X0


def project_points(points3d, P, K=None, dist=None):
    '''
    Project 3D points (Nx3 array) to image coordinates, given the projection matrix P (4x3 matrix)
    If K matric and dist vector are given, the function computes undistorted image projections (otherwise, zero distortions are assumed)
    '''
    points3d = cv2.convertPointsToHomogeneous(points3d)[:,0,:]
    m = np.matmul(P, points3d.T)
    m = m[0:2,:] / m[2,:]; 
    m = m.astype(float).T
    
    if dist is not None and K is not None:
        m = cv2.undistortPoints(m, K, dist, None, K)[:,0,:]
        
    return m.astype(float)