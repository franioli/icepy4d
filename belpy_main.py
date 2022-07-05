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

import numpy as np
import os
from pathlib import Path
import cv2
import pickle
import json
import matplotlib.pyplot as plt
import open3d as o3d
import pydegensac

from lib.classes import (Camera, Imageds, Features, Targets)
from lib.match_pairs import match_pair
from lib.track_matches import track_matches
from lib.io import read_img
from lib.geometry import estimate_pose
from lib.utils import (undistort_image, interpolate_point_colors, 
                       build_dsm, DSM, generate_ortophoto,
                       )
from lib.visualization import (draw_epip_lines, make_matching_plot)
from lib.point_clouds import (create_point_cloud, display_pc_inliers)
from lib.misc import (convert_to_homogeneous, 
                      convert_from_homogeneous,
                      )
from lib.thirdParts.triangulation import (linear_LS_triangulation, iterative_LS_triangulation)
from lib.thirdParts.transformations import affine_matrix_from_points

#---  Parameters  ---#
# TODO: put parameters in parser or in json file

#TODO: use Pathlib instead of strings and os
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

#- Bounding box for processing the images from the two cameras
# maskBB = [[600,1900,5300, 3600], [800,1800,5500,3500]] 
maskBB = [[400,1500,5500,4000], [600,1400,5700,3900]]

# Epoches to process
# It can be 'all' for processing all the epochs or a list with the epoches to be processed
epoches_to_process = [0,1,2,3,4,5,6,7] # 

# On-Off switches
find_matches = False


#--- Perform matching and tracking ---# 

#  Inizialize Lists
cameras =  [[] for x in range(numCams)] # List for storing cameras objects 
images = []   # List for storing image paths
F_matrix = [] # List for storing fundamental matrixes
points3d = [] # List for storing 3D points
features = [[] for x in range(numCams)] # List for storing all the matched features at all epochs

#- Create Image Datastore objects
for jj, cam in enumerate(camNames):
    images.append(Imageds(os.path.join(rootDirPath, imFld, cam)))
    if len(images[jj]) is not len(images[jj-1]):
        print('Error: different number of images per camera')
        # exit(1)        
print('Data loaded')


# epoch = 0
if find_matches:
    if epoches_to_process == 'all':
        epoches_to_process = [x for x in range(len(images[0]))]
    if epoches_to_process is None:
        print('Invalid input of epoches to process')
        exit(1)
    for epoch in epoches_to_process:
        print(f'Processing epoch {epoch}...')

        #=== Find Matches at current epoch ===#
        print(f'Run Superglue to find matches at epoch {epoch}')    
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
            print(f'Track points from epoch {epoch-1} to epoch {epoch}')
            
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
Notes 
'''
#TODO: put parameters, swithches and pre-processing all togheter at the beginning.
    
# Parameters
target_paths = [Path('data/target_image_p2.txt'), Path('data/target_image_p3.txt')] 

# On-Off switches
do_viz = False
rotate_ptc = False
do_coregistration = False    # If set to false, fix the camera EO as the first epoch
do_SOR_filter = True

# Iizialize variables
cameras = [[] for x in range(numCams)]
pcd = []
tform = []
img0 = images[0][0]
h, w, _ = img0.shape

# Build reference camera objects with only known interior orientation
ref_cams = []
for jj, cam in enumerate(camNames):
    calib_path = os.path.join(rootDirPath, calibFld, cam+'.txt')
    ref_cams.append(Camera(calib_path = calib_path))
   
# Read target image coordinates
targets = Targets(cam_id=[0,1],  im_coord_path=target_paths)

# Camera baseline
# TODO: put in a json file for general configuration
X01_meta = np.array([416651.5248,5091109.9121,1858.9084])   # IMG_2092
X02_meta = np.array([416622.2755,5091364.5071,1902.4053])   # IMG_0481
camWorldBaseline = np.linalg.norm(X01_meta - X02_meta)      # [m] From Metashape model in July

#--- SfM ---#
for epoch in epoches_to_process:
# epoch = 0
    # Initialize Intrinsics
    cameras[0].append(Camera(K=ref_cams[0].K, dist=ref_cams[0].dist))
    cameras[1].append(Camera(K=ref_cams[1].K, dist=ref_cams[1].dist))
          
    # Estimate Realtive Pose with Essential Matrix
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
    cameras[1][epoch].t_from_R_and_X0()
    
    if do_coregistration:
        # Triangulate targets
        pts0_und = cv2.undistortPoints(targets.get_im_coord(0)[epoch], cameras[0][epoch].K, 
                                       cameras[0][epoch].dist, None, cameras[0][epoch].K)[:,0,:]
        pts1_und = cv2.undistortPoints(targets.get_im_coord(1)[epoch], cameras[1][epoch].K, 
                                       cameras[1][epoch].dist, None, cameras[1][epoch].K)[:,0,:]
        M, status = iterative_LS_triangulation(pts0_und, cameras[0][epoch].P,  pts1_und, cameras[1][epoch].P)
        targets.append_obj_cord(M)
    
        # Estimate rigid body transformation between first epoch RS and current epoch RS 
        v0 = np.concatenate((cameras[0][0].X0, cameras[1][0].X0, 
                             targets.get_obj_coord()[0].reshape(3,1)), axis = 1)
        v1 = np.concatenate((cameras[0][epoch].X0, cameras[1][epoch].X0, 
                             targets.get_obj_coord()[epoch].reshape(3,1)), axis = 1)
        tform.append(affine_matrix_from_points(v1, v0, shear=False, scale=False, usesvd=True))
        
    elif epoch > 0:
        # Fix the EO of both the cameras as those estimated in the first epoch
        cameras[0][epoch] =  cameras[0][0]
        cameras[1][epoch] =  cameras[1][0]
    
    #--- Triangulate Points ---#
    #TODO: put in a separate function which includes undistortion and then triangulation
    pts0_und = cv2.undistortPoints(features[0][epoch].get_keypoints(), cameras[0][epoch].K, 
                                   cameras[0][epoch].dist, None, cameras[0][epoch].K)[:,0,:]
    pts1_und = cv2.undistortPoints(features[1][epoch].get_keypoints(), cameras[1][epoch].K, 
                                   cameras[1][epoch].dist, None, cameras[1][epoch].K)[:,0,:]
    points3d, status = iterative_LS_triangulation(pts0_und, cameras[0][epoch].P,  pts1_und, cameras[1][epoch].P)
    print(f'Point triangulation succeded: {status.sum()/status.size}')

    # Interpolate colors from image 
    jj = 1
    image = cv2.cvtColor(images[jj][epoch], cv2.COLOR_BGR2RGB) 
    #TODO: include color conversion in function interpolate_point_colors
    points3d_cols =  interpolate_point_colors(points3d, image, cameras[jj][epoch].P, 
                                              cameras[jj][epoch].K, cameras[jj][epoch].dist)
    print(f'Color interpolated on image {jj} ')
    
    if do_coregistration:    
        # Apply rigid body transformation to triangulated points
        pts = np.dot(tform[epoch], convert_to_homogeneous(points3d.T)) 
        points3d = convert_from_homogeneous(pts).T
        
    # Create point cloud and save .ply to disk
    pcd_epc = create_point_cloud(points3d, points3d_cols, 
                                  path=f'res/pt_clouds/sparse_pts_t{epoch}.ply')
    
    # Filter outliers in point cloud with SOR filter
    if do_SOR_filter:
        cl, ind = pcd_epc.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
        if do_viz:
            display_pc_inliers(pcd_epc, ind)
        pcd_epc =  pcd_epc.select_by_index(ind)   
        print("Point cloud filtered by Statistical Oulier Removal")
            
    # Perform rotation of 180deg around X axis   
    if rotate_ptc:
        ang = np.pi
        Rx = o3d.geometry.Geometry3D.get_rotation_matrix_from_axis_angle(np.array([1., 0., 0.], 
                                                                                  dtype='float64')*ang)
        pcd_epc.rotate(Rx)
    
    # Visualize point cloud    
    if do_viz:
        win_name = f'Sparse point cloud - Epoch: {epoch} - Num pts: {len(np.asarray(pcd[epoch].points))}'
        o3d.visualization.draw_geometries([pcd_epc], window_name=win_name, 
                                          width=1280, height=720, 
                                          left=300, top=200)    
    # Store point cloud in Point Cloud List     
    pcd.append(pcd_epc)
        
# Visualize all points clouds together
o3d.visualization.draw_geometries(pcd, window_name='All epoches', 
                                    width=1280, height=720, 
                                    left=300, top=200)

#%% tmp

# cam = 0
# for i in range(len(images[0])):
#     xy = targets.get_im_coord(cam)[i]
#     # x2,y2 = targets.get_im_coord(2,i)
#     img = images[cam][i]
#     cv2.namedWindow(f'cam {cam}, epoch {i}', cv2.WINDOW_NORMAL)
#     color=(0,255,0)
#     point_size=2
#     img_target = cv2.drawMarker(img,tuple(xy.astype(int)),color,cv2.MARKER_CROSS,1)
#     cv2.imshow(f'cam {cam}, epoch {i}', img_target) 
#     cv2.waitKey()
#     cv2.destroyAllWindows()


#%% DSM 
res = 0.03
dsms = []
for epoch in epoches_to_process:
    dsms.append(build_dsm(np.asarray(pcd[epoch].points), 
                            dsm_step=res, 
                            make_dsm_plot=False, 
                            save_path=f'sfm/dsm_approx_epoch_{epoch}.tif'
                            ))
print('DSM generated for all the epoches')
#TODO: implement better DSM class

# Generate Ortophotos 
jj = 0
epoch = 0
ortofoto = []
for jj in range(numCams):
    ortofoto.append(generate_ortophoto(cv2.cvtColor(images[jj][epoch], cv2.COLOR_BGR2RGB),
                                      dsms[epoch], cameras[jj][epoch],
                                      save_path=f'sfm/ortofoto_approx_cam_{jj}_epc_{epoch}.tif',
                                      ))
fig, ax = plt.subplots()
ax.imshow(ortofoto[1])

#%% 


#%% DENSE MATCHING


# Init
epoch = 0
sgm_path = Path('sgm')
downsample = 0.25
fast_viz = True

stem0, stem1 = images[0].get_image_stem(epoch), images[1].get_image_stem(epoch)
img0, img1 = images[0][epoch], images[1][epoch]
h, w, _ = img0.shape

# pts0, pts1 = features[0][epoch].get_keypoints(), features[1][epoch].get_keypoints()
# F, inlMask = pydegensac.findFundamentalMatrix(pts0, pts1, px_th=1, conf=0.99999,
#                                               max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson',
#                                               symmetric_error_check=True, enable_degeneracy_check=True)


#--- Rectify calibrated ---#
left_cam = cameras[1][epoch]
right_cam = cameras[0][epoch]
left_img = images[1][epoch]
rigth_img = images[0][epoch]


#       
rectOut = cv2.stereoRectify(left_cam.K, left_cam.dist, 
                            right_cam.K, right_cam.dist, 
                            (w,h), left_cam.R, left_cam.t, 
                            # (w,h), left_cam.R.T, -left_cam.t,                             
                            # (w,h), right_cam.R.T, -right_cam.t,                             
                            flags=cv2.CALIB_ZERO_DISPARITY,
                            )
# R1,R2,P1,P2,Q,validRoi0,validRoi1 
R0, R1 = rectOut[0], rectOut[1]
P0, P1 = rectOut[2], rectOut[3]
Q = rectOut[4]

# r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(k_left, d_left, k_right, d_right,
#                                                   (img_height, img_width),
#                                                   R,T,flags=cv2.CALIB_ZERO_DISPARITY)

map0x, map0y = cv2.initUndistortRectifyMap(cameraMatrix=left_cam.K, 
                                           distCoeffs=left_cam.dist,
                                           R=R0,
                                           newCameraMatrix=P0,
                                           size=(w,h),
                                           m1type=cv2.CV_32FC1
                                           )
map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix=right_cam.K, 
                                           distCoeffs=right_cam.dist,
                                           R=R1,
                                           newCameraMatrix=P1,
                                           size=(w,h),
                                           m1type=cv2.CV_32FC1
                                           )
img0_rect = cv2.remap(left_img, map0x, map0y, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT)
img1_rect = cv2.remap(rigth_img, map1x, map1y, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT)
cv2.imwrite(str(sgm_path / (stem0 + "_rectified.jpg")), img0_rect)
cv2.imwrite(str(sgm_path / (stem1 + "_rectified.jpg")), img1_rect)


# cv2.imshow('img0 rect', img0_rect)
# cv2.waitKey()
# cv2.destroyAllWindows()
    
# img0_rectified = cv2.warpPerspective(img0, R0, (w,h))
# img1_rectified = cv2.warpPerspective(img1, R1, (w,h))

# # write images to disk

# cv2.imwrite(path0, img0_rectified)
# cv2.imwrite(path1, img1_rectified)


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
