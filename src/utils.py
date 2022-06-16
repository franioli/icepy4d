import cv2
import numpy as np
import scipy as sp
from src.geometry import (P_from_KRT, project_points)

def normalize_and_und_points(pts, K, dist=None):
    pts = cv2.undistortPoints(pts.T, K, dist)
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
        cv2.imwrite(out_name, und)
    return und, K_scaled

def interpolate_point_colors(pointxyz, image, K, R, t, dist=None, winsz=1):
    ''''
    Interpolate color of a 3D sparse point cloud, given an oriented image
      Inputs:  
       - Nx3 matrix with 3d world points coordinates
       - image
       - camera interior and exterior orientation matrixes: K, R, t
       - distortion vector according to OpenCV
    Output: Nx3 colour matrix, as float numbers (normalized in [0,1])
    '''
    
    assert K is not None, 'invalid camera matrix' 
    assert R is not None, 'invalid rotation matrix' 
    assert t is not None, 'invalid translation vector' 
    assert image.ndim == 3, 'invalid input image. Image has not 3 channel'

    if K is not None and dist is not None:
        image = cv2.undistort(image, K, dist, None, K)
    
    numPts = len(pointxyz)
    col = np.zeros((numPts,3))
    h,w,_ = image.shape
    P = P_from_KRT(K, R, t)
    m = project_points(pointxyz, P, K, dist)
    image = image.astype(np.float32) / 255.
    
    for k in range(0,numPts):
        kint = np.round(m[k,0:2]).astype(int)
        i = np.array([a for a in range(kint[1]-winsz,kint[1]+winsz+1)])
        j = np.array([a for a in range(kint[0]-winsz,kint[0]+winsz+1)])
        if i.min()<0 or  i.max()>w or j.min()<0 or j.max()>h:
            continue
        ii, jj = np.meshgrid(i,j)
        ii, jj = ii.flatten(), jj.flatten()
        for rgb in range(0,3):
            colPatch = image[i[0]:i[-1]+1,j[0]:j[-1]+1,rgb]
            fcol = sp.interpolate.interp2d(i, j, colPatch, kind='linear')  
            col[k,rgb] = fcol(m[k,0], m[k,1])
    return col

## Visualization
def draw_epip_lines(img0, img1, lines, pts0, pts1, fast_viz=True):
    ''' img0 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img0.shape
    if not fast_viz:
        img0 = cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        #TODO: implement visualization in matplotlib
    for r,pt0,pt1 in zip(lines,pts0,pts1):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img0 = cv2.line(img0, (x0,y0), (x1,y1), color,1)
        img0 = cv2.circle(img0,tuple(pt0.astype(int)),5,color,-1)
        img1 = cv2.circle(img1,tuple(pt1.astype(int)),5,color,-1)
        # img0 = cv2.drawMarker(img0,tuple(pt0.astype(int)),color,cv2.MARKER_CROSS,3)
        # img1 = cv2.drawMarker(img1,tuple(pt1.astype(int)),color,cv2.MARKER_CROSS,3)        
    return img0,img1


def make_matching_plot(image0, image1, pts0, pts1, pts_col=(0,0,255), point_size=2, line_col=(0,255,0), line_thickness=1, path=None, margin=10):
    if image0.ndim > 2:
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    if image1.ndim > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)    
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    mkpts0, mkpts1 = np.round(pts0).astype(int), np.round(pts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=line_col, thickness=line_thickness, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), point_size, pts_col, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), point_size, pts_col, -1,
                   lineType=cv2.LINE_AA)
    if path is not None: 
        cv2.imwrite(path, out)
        
        
