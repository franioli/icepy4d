from pathlib import Path
import numpy as np
import cv2
import pydegensac
# import scipy.linalg as alg

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.9999):
    """
    Estimate camera pose given matched points and intrinsics matrix.
    Function taken by the code of SuperGlue, by Sarlin et al., 2020
    https://github.com/magicleap/SuperGluePretrainedNetwork
    """
    if len(kpts0) < 5:
        return None

    # f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    # norm_thresh = thresh / f_mean

    # kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    # kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # E, mask = cv2.findEssentialMat(
    #     kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
    #     method=cv2.RANSAC)

    # assert E is not None
    
    # best_num_inliers = 0
    # ret = None
    # for _E in np.split(E, len(E) / 3):
    #     n, R, t, _ = cv2.recoverPose(
    #         _E, nkpts0, nkpts1, np.eye(3), 1e9, mask=mask)
    #     if n > best_num_inliers:
    #         best_num_inliers = n
    #         ret = (R, t[:, 0], mask.ravel() > 0)
    # return ret
    
    
    F, mask = pydegensac.findFundamentalMatrix( kpts0, kpts1, px_th=thresh, conf=conf, 
                                                max_iters=10000, laf_consistensy_coef=-1.0, 
                                                error_type='sampson', symmetric_error_check=True, 
                                                enable_degeneracy_check=True)
    E = np.dot(K1.T, np.dot(F, K0))

    kpts0 = cv2.convertPointsToHomogeneous(kpts0)[:,0,:].T
    kpts1 = cv2.convertPointsToHomogeneous(kpts1)[:,0,:].T
    nkpts0 = np.dot(np.linalg.inv(K0), kpts0)
    nkpts0 /= nkpts0[2,:]
    nkpts0 = nkpts0[0:2,:].T
    nkpts1 = np.dot(np.linalg.inv(K1), kpts1)    
    nkpts1 /= nkpts1[2,:]
    nkpts1 = nkpts1[0:2,:].T

    nkpts0 = nkpts0[mask,:]
    nkpts1 = nkpts1[mask,:]  

    n, R, t, _  = cv2.recoverPose(E, nkpts0, nkpts1, np.eye(3), 1e9)
    ret = (R, t[:, 0], mask)
    
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
        

def scale_intrinsics(K, scales):
    """ 
    Scale camera intrisics matrix (K) after image downsampling or upsampling.
    """
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

def C_from_P(P):
    """
    Compute camera perspective centre from 3x4 P matrix, as:
        C = - inv(KR) * Kt 
    """
    C = -np.matmul(np.linalg.inv(P[:,0:3]),P[:,3]).reshape(3,1)
    return C


def project_points(points3d, P, K=None, dist=None):
    '''
    Project 3D points (Nx3 array) to image coordinates, given the projection matrix P (4x3 matrix)
    If K matric and dist vector are given, the function computes undistorted image projections (otherwise, zero distortions are assumed)
    Returns: 2D projected points (Nx2 array) in image coordinates
    '''
    points3d = cv2.convertPointsToHomogeneous(points3d)[:,0,:]
    m = np.matmul(P, points3d.T)
    m = m[0:2,:] / m[2,:]; 
    m = m.astype(float).T
    
    if dist is not None and K is not None:
        m = cv2.undistortPoints(m, K, dist, None, K)[:,0,:]
        
    return m.astype(float)
