import numpy as np
import os
from pathlib import Path
import cv2 
import pickle
import json 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d as o3d
import pydegensac

from lib.classes import (Camera, DSM, Features, Imageds)
from lib.match_pairs import match_pair
from lib.track_matches import track_matches
from lib.io import read_img
from lib.geometry import (estimate_pose, P_from_KRT, X0_from_P, project_points)
from lib.utils import (undistort_image, interpolate_point_colors, build_dsm)
from lib.visualization import (draw_epip_lines, make_matching_plot)
from lib.thirdParts.triangulation import (linear_LS_triangulation, iterative_LS_triangulation)

#---  Parameters  ---#
# TODO: put parameters in parser or in json file

rootDirPath = '.'

#- Folders and paths
imFld = 'data/img'
imExt = '.tif'
calibFld = 'data/calib'
matching_config = 'config/opt_matching.json'
tracking_config = 'config/opt_tracking.json'

#- CAMERAS
numCams = 2
camNames = ['p2', 'p3']

#- Image cropping boundaries
# maskBB = [[600,1900,5300, 3600], [800,1800,5500,3500]]             # Bounding box for processing the images from the two cameras
maskBB = [[400,1500,5500,4000], [600,1400,5700,3900]]             # Bounding box for processing the images from the two cameras

#  Inizialize Lists
cameras = [[],[]]  # List for storing cameras information (as dicts)
images = []   # List for storing image paths
F_matrix = [] # List for storing fundamental matrixes
points3d = [] # List for storing 3D points
features = [[],[]]  # List for storing all the matched features at all epochs

#- images
for jj, cam in enumerate(camNames):
    images.append(Imageds(os.path.join(rootDirPath, imFld, cam)))  
    if len(images[jj]) is not len(images[jj-1]):
        print('Error: different number of images per camera')
        # exit(1)# Remove some variables
        del data, K, dist, path, f,  jj

print('Data loaded')

#%% Perform matching and tracking
find_matches = 0
epoches2process = [0,1,2,3,4,5,6,7] # #1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
epoch = 0


if find_matches:
    for epoch in epoches2process:
        print(f'Processing epoch {epoch}...')

        #=== Find Matches at current epoch ===#
        print('Run Superglue to find matches at epoch {}'.format(epoch))    
        epochdir = os.path.join('res','epoch_'+str(epoch))      
        with open(matching_config,) as f:
            opt_matching = json.load(f)
        opt_matching['output_dir'] = epochdir
        pair = [images[0].get_image_path(epoch), images[1].get_image_path(epoch)]
        maskBB = np.array(maskBB).astype('int')
        matchedPts, matchedDescriptors, matchedPtsScores = match_pair(pair, maskBB, opt_matching)

        # Store matches in features structure
        for jj in range(numCams):
            #TODO: add better description
            ''' 
            External list are the cameras, internal list are the epoches... 
            '''
            features[jj].append(Features())
            features[jj][epoch].append_features({'kpts': matchedPts[jj],
                                                'descr': matchedDescriptors[jj], 
                                                'score': matchedPtsScores[jj]})
            # TODO: Store match confidence!

        #=== Track previous matches at current epoch ===#
        if epoch > 0:
            print('Track points from epoch {} to epoch {}'.format(epoch-1, epoch))
            
            trackoutdir = os.path.join('res','epoch_'+str(epoch), 'from_t'+str(epoch-1))
            with open(tracking_config,) as f:
                opt_tracking = json.load(f)
            opt_tracking['output_dir'] = trackoutdir
            pairs = [ [ images[0].get_image_path(epoch-1), images[0].get_image_path(epoch)], 
                      [ images[1].get_image_path(epoch-1), images[1].get_image_path(epoch)] ] 
            prevs = [features[0][epoch-1].get_features_as_dict(),
                     features[1][epoch-1].get_features_as_dict()]
            tracked_cam0, tracked_cam1 = track_matches(pairs, maskBB, prevs, opt_tracking)
            # TODO: tenere traccia dell'epoca in cui è stato trovato il match
            # TODO: Check bounding box in tracking
            # TODO: clean tracking code

            # Store all matches in features structure
            features[0][epoch].append_features( {'kpts': tracked_cam0['keypoints1'],
                                                 'descr': tracked_cam0['descriptors1'] , 
                                                 'score': tracked_cam0['scores1']} ) 
            features[1][epoch].append_features( {'kpts': tracked_cam1['keypoints1'],
                                                 'descr': tracked_cam1['descriptors1'] , 
                                                 'score': tracked_cam1['scores1']} )
            
            # Run Pydegensac to estimate F matrix and reject outliers                         
            F, inlMask = pydegensac.findFundamentalMatrix(features[0][epoch].get_keypoints(), features[1][epoch].get_keypoints(), 
                                                          px_th=2, conf=0.9, max_iters=100000, laf_consistensy_coef=-1.0, 
                                                          error_type='sampson', symmetric_error_check=True, enable_degeneracy_check=True)
            F_matrix.append(F)
            print('Matches at epoch {}: pydegensac found {} inliers ({:.2f}%)'.format(epoch, inlMask.sum(),
                            inlMask.sum()*100 / len(features[0][epoch]) ))

        # Write matched points to disk   
        im_stems = images[0].get_image_stem(epoch), images[1].get_image_stem(epoch)
        for jj in range(numCams):
            features[jj][epoch].save_as_txt(os.path.join(epochdir, im_stems[jj]+'_mktps.txt'))
        with open(os.path.join(epochdir, im_stems[0]+'_'+im_stems[1]+'_features.pickle'), 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Matching completed')

elif not features[0]: 
    last_match_path = 'res/epoch_7/IMG_0541_IMG_2152_features.pickle'
    with open(last_match_path, 'rb') as f:
        features = pickle.load(f)
        print("Loaded previous matches")
else:
    print("Features already present, nothing was changed.")


#%% SfM
'''
Note t
'''
cameras = [[],[]]  # List for storing cameras information (as dicts)
points3d = []
pcd = []
do_viz = False
rotate_ptc = False

K_calib, dist_calib = [],[]
#- Cameras structures
for jj, cam in enumerate(camNames):
    with open(os.path.join(rootDirPath, calibFld, cam+'.txt'), 'r') as f:
        data = np.loadtxt(f)
    K_calib.append(data[0:9].astype(float).reshape(3, 3, order='C'))
    dist_calib.append(data[9:13].astype(float))
del data, f,  jj

# Camera baseline
X01_meta = np.array([416651.5248,5091109.9121,1858.9084])   # IMG_2092
X02_meta = np.array([416622.2755,5091364.5071,1902.4053]) # IMG_0481
camWorldBaseline = np.linalg.norm(X01_meta - X02_meta)                         # [m] From Metashape model at epoch t0

img0 = images[0][0]
h, w, _ = img0.shape

#--- Realtive Pose with Essential Matrix ---#
for epoch in epoches2process:
# epoch = 0
    # Initialize Intrinsics
    cameras[0].append(Camera(K=K_calib[0], dist=dist_calib[0]))
    cameras[1].append(Camera(K=K_calib[1], dist=dist_calib[1]))
   
    # FIX SECOND CAMERA TO EO OF FIRST EPOCH!!
    #TODO: leave second camera free to move and estimate rototranslation  of RS
    if epoch == 0:
        
        # Estimate Relativa Orientation
        pts0, pts1 = features[0][epoch].get_keypoints(), features[1][epoch].get_keypoints()
        rel_pose = estimate_pose(pts0, pts1, cameras[0][epoch].K,  cameras[1][epoch].K, thresh=1, conf=0.9999)
        R, t, valid = rel_pose[0], rel_pose[1], rel_pose[2]
        print('Computing relative pose. Valid points: {}/{}'.format(valid.sum(),len(valid)))
        
        # Update cameras structures
        cameras[1][epoch].R, cameras[1][epoch].t = R, t.reshape(3,1)
        cameras[1][epoch].compose_P()
        cameras[1][epoch].camera_center()
            
        # Scale model by using camera baseline
        camRelOriBaseline = np.linalg.norm(cameras[0][epoch].X0 - cameras[1][epoch].X0)
        scaleFct = camWorldBaseline / camRelOriBaseline
        
        # Update Camera EO
        cameras[1][epoch].X0 = cameras[1][epoch].X0 * scaleFct
        cameras[1][epoch].t = -np.dot(cameras[1][epoch].R, cameras[1][epoch].X0)
        cameras[1][epoch].compose_P()
        
    else:
        
        cameras[1][epoch].R = cameras[1][0].R
        cameras[1][epoch].t = cameras[1][0].t
        cameras[1][epoch].X0 = cameras[1][0].X0
        cameras[1][epoch].P = cameras[1][0].P
        
    #--- Triangulate Points ---#
    pts0_und = cv2.undistortPoints(features[0][epoch].get_keypoints(), cameras[0][epoch].K, 
                                   cameras[0][epoch].dist, None, cameras[0][epoch].K)[:,0,:]
    pts1_und = cv2.undistortPoints(features[1][epoch].get_keypoints(), cameras[1][epoch].K, 
                                   cameras[1][epoch].dist, None, cameras[1][epoch].K)[:,0,:]
    M, status = iterative_LS_triangulation(pts0_und, cameras[0][epoch].P,  pts1_und, cameras[1][epoch].P)
    points3d.append(M)
    print(f'Point triangulation succeded: {status.sum()/status.size}')
    
    # Interpolate colors from image 
    jj = 1
    image = cv2.cvtColor(images[jj][epoch], cv2.COLOR_BGR2RGB)
    points3d_cols =  interpolate_point_colors(points3d[epoch], image, cameras[jj][epoch].P, 
                                              cameras[jj][epoch].K, cameras[jj][epoch].dist)
    print(f'Color interpolated on image {jj} ')
 
    # Visualize and export sparse point cloud
    pcd.append(o3d.geometry.PointCloud())
    pcd[epoch].points = o3d.utility.Vector3dVector(points3d[epoch])
    pcd[epoch].colors = o3d.utility.Vector3dVector(points3d_cols)
    
    # Save point cloud to disk
    o3d.io.write_point_cloud("res/pt_clouds/sparse_ptd_t"+str(epoch)+".ply", pcd[epoch])
    
    if rotate_ptc:
        ang =np.pi
        Rx = o3d.geometry.Geometry3D.get_rotation_matrix_from_axis_angle(np.array([1., 0., 0.], dtype='float64')*ang)
        pcd[epoch].rotate(Rx)
        
    if do_viz:
        win_name = f'Sparse point cloud - Epoch: {epoch} - Num pts: {len(np.asarray(pcd[epoch].points))}'
        o3d.visualization.draw_geometries([pcd[epoch]], window_name=win_name, 
                                          width=1280, height=720, 
                                          left=300, top=200)
    

# Visualize all points clouds together
o3d.visualization.draw_geometries(pcd, window_name='All epoches ', 
                                    width=1280, height=720, 
                                    left=300, top=200)

#%% BUILD DSM AND ORTHOPHOTOS

## Build approximate DSM
dsm = build_dsm(points3d[epoch], dsm_step=0.1, save_path="sfm/dsm_approx.tif")

# Generate Ortophotos
from src.geometry import (P_from_KRT, project_points)

def generate_ortophoto(image, dsm, P, res=1):
    xx = dsm.x
    yy = dsm.y
    zz = dsm.z
    
    dsm_shape = dsm.x.shape
    ncell = dsm_shape[0]*dsm_shape[1]
    xyz = np.zeros((ncell,3))
    xyz[:,0] = xx.flatten()
    xyz[:,1] = yy.flatten()
    xyz[:,2] = zz.flatten()
    valid_cell = np.invert(np.isnan(xyz[:,2]))
    
    cols = np.full((ncell,3),0)
    cols[valid_cell,:] = interpolate_point_colors(xyz[valid_cell,:], image, cameras.P, cameras.K, cameras.dist)
    ortophoto = np.zeros((dsm_shape[0],dsm_shape[1],3))
    ortophoto[:,:,0] = cols[:,0].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto[:,:,1] = cols[:,1].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto[:,:,2] = cols[:,2].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto = np.uint8(ortophoto*255)
    
    # for a, b, c in zip(xx.flatten(), yy.flatten(), zz.flatten()):
    #     xyz.append([a, b, c]) 
    # ortophoto = cols[:,0].reshape(dsm_shape[0], dsm_shape[1],3)
    # ortophoto = np.uint8(ortophoto*255)

    return xyz, cols, ortophoto

    # ortophoto = None
    # return ortophoto


#%% DENSE MATCHING

# Init
epoch = 0
sgm_path = Path('sgm')
downsample = 0.25
fast_viz = True

stem0, stem1 = images[0].get_image_stem(epoch), images[1].get_image_stem(epoch)

pts0, pts1 = features[0][epoch].get_keypoints(), features[1][epoch].get_keypoints()
F, inlMask = pydegensac.findFundamentalMatrix(pts0, pts1, px_th=1, conf=0.99999,
                                              max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson',
                                              symmetric_error_check=True, enable_degeneracy_check=True)
img0, img1 = images[0][epoch], images[1][epoch]
h, w, _ = img0.shape

#--- Rectify calibrated ---#
rectOut = cv2.stereoRectify(cameras[0][epoch].K, cameras[0][epoch].dist, 
                            cameras[1][epoch].K, cameras[1][epoch].dist, 
                            (h,w), cameras[1][epoch].R, cameras[1][epoch].t)
# R1,R2,P1,P2,Q,validRoi0,validRoi1 
R0, R1 = rectOut[0], rectOut[1]
P0, P1 = rectOut[2], rectOut[3]
Q = rectOut[4]
    
img0_rectified = cv2.warpPerspective(img0, R0, (w,h))
img1_rectified = cv2.warpPerspective(img1, R1, (w,h))

# write images to disk
path0 = str(sgm_path / (stem0 + "_rectified.jpg"))
path1 = str(sgm_path / (stem1 + "_rectified.jpg"))
cv2.imwrite(path0, img0_rectified)
cv2.imwrite(path1, img1_rectified)


#--- Run PSMNet to compute disparity ---#


#%%
#--- Recify Uncalibrated ---#

# Udistort images
img0, img1 = images[1][epoch], images[0][epoch]
name0 = str(sgm_path / 'und' / (stem0 + "_undistorted.jpg"))
name1 = str(sgm_path / 'und' / (stem1 + "_undistorted.jpg"))
img0, K0_scaled = undistort_image(img0, cameras[0][epoch].K, cameras[0][epoch].dist, downsample, name0)
img1, K1_scaled = undistort_image(img1, cameras[1][epoch].K, cameras[1][epoch].dist, downsample, name1)
h, w, _ = img0.shape

# Undistot points and compute F matrx
pts0 = features[0][epoch].get_keypoints()*downsample
pts1 = features[1][epoch].get_keypoints()*downsample
pts0 = cv2.undistortPoints(pts0, cameras[0][epoch].K, cameras[0][epoch].dist, None, cameras[0][epoch].K)
pts1 = cv2.undistortPoints(pts1, cameras[1][epoch].K, cameras[1][epoch].dist, None, cameras[1][epoch].K)
F, inlMask = pydegensac.findFundamentalMatrix(pts0, pts1, px_th=1, conf=0.99999,
                                              max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson',
                                              symmetric_error_check=True, enable_degeneracy_check=True)
print('Pydegensac: {} inliers ({:.2f}%)'.format(inlMask.sum(), inlMask.sum()*100 / len(pts0)))

# Rectify uncalibrated
success, H1, H0 = cv2.stereoRectifyUncalibrated(pts0, pts1 , F, (w,h))
img0_rectified = cv2.warpPerspective(img0, H0, (w,h))
img1_rectified = cv2.warpPerspective(img1, H1, (w,h))

# write images to disk
path0 = str(sgm_path / 'rectified' / (stem0 + "_rectified.jpg"))
path1 = str(sgm_path / 'rectified' / (stem1 + "_rectified.jpg"))
cv2.imwrite(path0, img0_rectified)
cv2.imwrite(path1, img1_rectified)

if fast_viz: 
    print()
else:
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img0_rectified, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2RGB))
    plt.show()
    
    
#%%


#--- Find epilines corresponding to points in right image (second image) and drawing its lines on left image ---#
img0, img1 = images[0][0], images[1][0]

lines0 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 2, F)
lines0 = lines0.reshape(-1,3)
img0_epiplines, _ = draw_epip_lines(img0,img1,lines0,pts0,pts1)

# Find epilines corresponding to points in left image (first image) and drawing its lines on right image
lines1 = cv2.computeCorrespondEpilines(pts0.reshape(-1,1,2), 1, F)
lines1 = lines1.reshape(-1,3)
img1_epiplines,_ = draw_epip_lines(img1,img0,lines1,pts1,pts0, fast_viz=True)

if fast_viz:
    cv2.imwrite(str(sgm_path / (stem0 + "_epiplines.jpg")), img0_epiplines)
    cv2.imwrite(str(sgm_path / (stem1 + "_epiplines.jpg")), img1_epiplines)
else: 
    plt.subplot(121),plt.imshow(img0_epiplines)
    plt.subplot(122),plt.imshow(img1_epiplines)
    plt.show()
    
#--- Draw keypoints and matches ---#
pts0_rect = cv2.perspectiveTransform(np.float32(pts0).reshape(-1,1,2), H0).reshape(-1,2)
pts1_rect = cv2.perspectiveTransform(np.float32(pts1).reshape(-1,1,2), H1).reshape(-1,2)

# img0_rect_kpts = img0.copy()
img0 = cv2.cvtColor(images[0][0], cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(images[1][0], cv2.COLOR_BGR2GRAY)
pts0, pts1 = features[0]['mkpts0'], features[0]['mkpts1']
img0_kpts = cv2.drawKeypoints(img0,cv2.KeyPoint.convert(pts0),img0,(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img1_kpts = cv2.drawKeypoints(img1,cv2.KeyPoint.convert(pts1),img1,(0,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('kpts0.jpg', img0_kpts)
cv2.imwrite('kpts1.jpg', img1_kpts)

make_matching_plot(img0, img1, pts0, pts1, path='matches.jpg')        
