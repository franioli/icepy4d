#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:52:00 2022

@author: photogrammetry
"""


def compute_reprojection_error_points(self,P,feat3D,view_feat2D):
  		'''
  		Method that returns the reprojection error for a cloud of 3D points.
  		Args: 
  			P: the projection matrix of the view
  			feat3D: the 3D point cloud coordinates
  			view_feat2D: 2D feature coordinates for all the 3D points in that view
  		Returns: 
  			the reprojection error
  		'''
  		reproj_feature=P.dot(np.transpose(feat3D))
  		reproj_feature=reproj_feature/reproj_feature[2,:]
  		error=sum(sum(np.power((view_feat2D-reproj_feature[0:2,:]),2)))
  		return error

#%%


#%%
from pathlib import Path
import numpy as np
import cv2
import pydegensac
import scipy.linalg as alg

kpts0, kpts1, K0, K1, w, h, thresh, conf = pts0, pts1, cameras[0][epoch].K, cameras[1][epoch].K, w, h, 1, 0.9999


F, mask = pydegensac.findFundamentalMatrix( kpts0, kpts1, px_th=thresh, conf=conf, 
                                        max_iters=10000, laf_consistensy_coef=-1.0, 
                                        error_type='sampson', symmetric_error_check=True, 
                                        enable_degeneracy_check=True)
# E = np.dot(K1.T, np.dot(F, K0))

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


E = np.dot(K1.T, np.dot(F, K0))

n, R, t, _  = cv2.recoverPose(E, nkpts0, nkpts1, np.eye(3), 1e9)
ret = (R, t[:, 0], mask)


f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
norm_thresh = thresh / f_mean

kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

E2, mask = cv2.findEssentialMat(
    kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
    method=cv2.RANSAC)

assert E is not None

best_num_inliers = 0
ret = None
for _E in np.split(E, len(E) / 3):
    n, R, t, _ = cv2.recoverPose(
        _E, nkpts0, nkpts1, np.eye(3), 1e9, mask=mask)
    if n > best_num_inliers:
        best_num_inliers = n
        ret = (R, t[:, 0], mask.ravel() > 0)
return ret
    
    





def self_calibrate(F, w, h):
	# Compute the semi-calibrated fundamental matrix
	K = np.array([[2 * (w + h), 0, w / 2], [0, 2 * (w + h), h / 2], [0, 0, 1]])
	G = normalize_norm(np.dot(K.T, np.dot(F, K)))

	# Self-calibration using the Kruppa equations (Sturm's method)
	U, s, Vh = alg.svd(G)
	fp = np.array([s[0]**2 * (1 - U[2, 0]**2) * (1 - Vh[0, 2]**2) - s[1]**2 * (1 - U[2, 1]**2) * (1 - Vh[1, 2]**2),
		s[0]**2 * (U[2, 0]**2 + Vh[0, 2]**2 - 2 * U[2, 0]**2 * Vh[0, 2]**2) - s[1]**2 * (U[2, 1]**2 + Vh[1, 2]**2 - 2 * U[2, 1]**2 * Vh[1, 2]**2),
		s[0]**2 * U[2, 0]**2 * Vh[0, 2]**2 - s[1]**2 * U[2, 1]**2 * Vh[1, 2]**2])

	rs = np.roots(fp)
	rs = np.real(rs[abs(np.imag(rs)) < 1e-6])
	rs = rs[rs > 0]
	
	f = 2 * (w + h)
	if any(abs(fp) > 1e-6) and len(rs) > 0:
		f = 2 * (w + h) * np.sqrt(rs[0])
	
	K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
	E = np.dot(K.T, np.dot(F, K))								# E = K.T * F * K
	return stabilize(normalize_norm(E)), K

def normalize_norm(A):
	return A / alg.norm(A)

def stabilize(x, tol = 1e-6):
	xs = x.copy()
	xs[abs(xs) < tol] = 0
	return xs