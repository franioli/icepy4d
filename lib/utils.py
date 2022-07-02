import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import (interp2d, griddata)
from PIL import Image

from lib.geometry import project_points
# from lib.classes import DSM

def normalize_and_und_points(pts, K=None, dist=None):
    #TODO: Remove function and create better one...
    pts = cv2.undistortPoints(pts, K, dist)
    return pts

def undistort_image(image, K, dist, downsample=1, out_path=None):
    '''
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

def interpolate_point_colors(pointxyz, image, P, K=None, dist=None, winsz=1):
    ''''
    Interpolate color of a 3D sparse point cloud, given an oriented image
      Inputs:  
       - Nx3 matrix with 3d world points coordinates
       - image
       - camera interior and exterior orientation matrixes: K, R, t
       - distortion vector according to OpenCV
    Output: Nx3 colour matrix, as float numbers (normalized in [0,1])
    '''
    
    assert P is not None, 'invalid projection matrix' 
    assert image.ndim == 3, 'invalid input image. Image has not 3 channel'

    if K is not None and dist is not None:
        image = cv2.undistort(image, K, dist, None, K)
    
    numPts = len(pointxyz)
    col = np.zeros((numPts,3))
    h,w,_ = image.shape
    projections = project_points(pointxyz, P, K, dist)
    image = image.astype(np.float32) / 255.
    
    for k, m in enumerate(projections):
        kint = np.round(m).astype(int)
        i = np.array([a for a in range(kint[1]-winsz,kint[1]+winsz+1)])
        j = np.array([a for a in range(kint[0]-winsz,kint[0]+winsz+1)])
        if i.min()<0 or i.max()>h or j.min()<0 or j.max()>w:
            continue
        ii, jj = np.meshgrid(i,j)
        ii, jj = ii.flatten(), jj.flatten()
        for rgb in range(0,3):
            colPatch = image[i[0]:i[-1]+1,j[0]:j[-1]+1,rgb]
            fcol = interp2d(i, j, colPatch, kind='linear')  
            col[k,rgb] = fcol(m[0], m[1])
    return col
        

class DSM:
    ''' Class to store and manage DSM. '''
    def __init__(self, x, y, z, res):
        xx, yy = np.meshgrid(x,y)
        self.x = xx
        self.y = yy
        self.z = z
        self.res = res    
        
def build_dsm(points3d, dsm_step=1, xlim=None, ylim=None, save_path=None, do_viz=0):
    assert np.any(np.array(points3d.shape) == 3), "Invalid size of input points"
    if points3d.shape[0] == points3d.shape[1]:
        print("Warning: input vector 3 points. Unable to check validity of point dimensions.")        
    if points3d.shape[0] == 3:
        points3d = points3d.T
    x, y, z = points3d[:,0], points3d[:,1], points3d[:,2]
    if xlim is None:
        xlim = [np.floor(x.min()), np.ceil(x.max())]
    if ylim is None:
        ylim = [np.floor(y.min()), np.ceil(y.max())]
    
    # Interpolate dsm
    xq = np.arange(xlim[0], xlim[1], dsm_step)
    yq = np.arange(ylim[0], ylim[1], dsm_step)
    xx, yy = np.meshgrid(xq,yq)
    z_range = [np.floor(z.min()), np.ceil(z.max())]
    zz = griddata((x,y) , z, (xx, yy))
    dsm = DSM(xx, yy, zz, dsm_step)

    # plot dsm 
    if do_viz:
        ax = plt.figure()
        im = plt.contourf(xx, yy, dsm.z)
        scatter = plt.scatter(x, y, 1, c='k', alpha=0.5, marker='.')
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(im)
        cbar.set_label("z")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        plt.savefig('dsm_approx_plt.png', bbox_inches='tight')

    # Save dsm as tif
    if save_path is not None:
        dsm_ras = Image.fromarray(dsm.z)
        dsm_ras.save(save_path)        
    
    return dsm