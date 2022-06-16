import cv2
import numpy as np
import scipy as sp

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

def interpolate_point_colors(pointsXYZ, image, K, R, t, dist=None, winsz=1, do_viz=False):
    ''''
    Interpolate color of a 3D sparse point cloud, given an oriented image
      Inputs:  
       - 3xN matrix with 3d world points coordinates
       - image
       - camera interior and exterior orientation matrixes: K, R, t
       - distortion vector according to OpenCV
    Output: 3xN colour matrix 
    '''
    
    assert K is not None, 'invalid camera matrix' 
    assert R is not None, 'invalid rotation matrix' 
    assert t is not None, 'invalid translation vector' 

    if K is not None and dist is not None:
        image = undistort_image(image, K, dist)
    
    numPts = len(pointsXYZ)
    col = np.zeros((3,numPts))
    h,w = image.shape
    m,_ = cv2.projectPoints(pointsXYZ, cv2.Rodrigues(R), t, K, dist)

    for k in range(0,numPts):
        kint = np.round(m[:,k])
        i = [a for a in range(kint[1]-winsz,kint[1]+winsz)]
        j = [b for b in range(kint[0]-winsz,kint[0]+winsz)]
        if i > w or j > h:
            continue
        ii, jj = np.meshgrid(i,j)
        for rgb in range(0,3):
            colNum  = image[i,j,rgb].astype(float64).flatten

#         for  rgb = 1:3
#             colNum  = double(im(i(:),j(:),rgb));
#             fcol    = scatteredInterpolant(jj(:),ii(:),colNum(:), 'linear');
#             col(rgb,k) = fcol(m(1,k), m(2,k));
   
#     col = uint8(col);

#     if do_viz == true
#         figure; imshow(im); hold on;  axis on; 
#         scatter(m(1,:), m(2,:), 20, double(col(:, :))'/255,  'filled' , 'MarkerEdgeColor', 	[0 0 0])

    


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
        
        
