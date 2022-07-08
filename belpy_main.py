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
import h5py
import json
import matplotlib.pyplot as plt
import open3d as o3d
import pydegensac

from lib.classes import (Camera, Imageds, Features, Targets)
from lib.match_pairs import match_pair
from lib.track_matches import track_matches
from lib.io import read_img
from lib.geometry import (estimate_pose,
                          undistort_image,
                          undistort_points,
                          project_points,
                          )
from lib.utils import (interpolate_point_colors,
                       build_dsm, 
                       generate_ortophoto,
                       DSM,
                       )
from lib.visualization import (make_camera_pyramid,
                               draw_epip_lines, 
                               make_matching_plot, 
                               )
from lib.point_clouds import (
    create_point_cloud, display_pc_inliers, write_ply)
from lib.misc import (convert_to_homogeneous,
                      convert_from_homogeneous,
                      create_directory,
                      )

from lib.thirdParts.triangulation import (linear_LS_triangulation, 
                                          iterative_LS_triangulation,
                                          )
from lib.thirdParts.transformations import affine_matrix_from_points
from lib.thirdParts.camera_pose_visualizer import CameraPoseVisualizer

#---  Parameters  ---#
# TODO: put parameters in parser or in json file
# TODO: use Pathlib instead of strings and os

#- Folders and paths
imFld = 'data/img'
imExt = '.tif'
calibFld = 'data/calib'
matching_config = 'config/opt_matching.json'
tracking_config = 'config/opt_tracking.json'
res_folder = 'res'

#- CAMERAS
numCams = 2
cam_names = ['p2', 'p3']

# - Bounding box for processing the images from the two cameras
# maskBB = [[600,1900,5300, 3600], [800,1800,5500,3500]]
maskBB = [[400, 1500, 5500, 4000], [600, 1400, 5700, 3900]]

# On-Off switches
find_matches = False

# Epoches to process
# It can be 'all' for processing all the epochs or a list with the epoches to be processed
epoches_to_process = [x for x in range(3)]  # 'all' #

#--- Perform matching and tracking ---#

#  Inizialize Variables
# TODO: replace all lists with dictionaries
# TODO: replace cam0, cam1 with iterable objects
cam0, cam1 = cam_names[0], cam_names[1]
images = dict.fromkeys(cam_names)  # List for storing image paths
features = dict.fromkeys(cam_names)  # List for storing image paths
F_matrix = []  # List for storing fundamental matrixes
points3d = []  # List for storing 3D points

# - Create Image Datastore objects
for jj, cam in enumerate(cam_names):
    images[cam] = Imageds(Path(imFld) / cam)

# Check that number of images is the same for every camera
if len(images[cam1]) is not len(images[cam0]):
    print('Error: different number of images per camera')
    raise SystemExit(0)
else:
    print('Image datastores created.')

# Define epoches to be processed
if epoches_to_process == 'all':
    epoches_to_process = [x for x in range(len(images[cam0]))]
if epoches_to_process is None:
    print('Invalid input of epoches to process')
    raise SystemExit(0)

# epoch = 0
if find_matches:
    features[cam0], features[cam1] = [], []

    for epoch in epoches_to_process:
        print(f'Processing epoch {epoch}...')

        #=== Find Matches at current epoch ===#
        print(f'Run Superglue to find matches at epoch {epoch}')
        epochdir = Path(res_folder) / f'epoch_{epoch}'
        with open(matching_config,) as f:
            opt_matching = json.load(f)
        opt_matching['output_dir'] = epochdir
        pair = [images[cam0].get_image_path(
            epoch), images[cam1].get_image_path(epoch)]
        maskBB = np.array(maskBB).astype('int')
        matchedPts, matchedDescriptors, matchedPtsScores = match_pair(
            pair, maskBB, opt_matching)

        # Store matches in features structure
        for jj, cam in enumerate(cam_names):
            # TODO: add better description
            ''' 
            External list are the cameras, internal list are the epoches... 
            '''
            features[cam].append(Features())
            features[cam][epoch].append_features({'kpts': matchedPts[jj],
                                                  'descr': matchedDescriptors[jj],
                                                  'score': matchedPtsScores[jj]})
            # TODO: Store match confidence!

        #=== Track previous matches at current epoch ===#
        if epoch > 0:
            print(f'Track points from epoch {epoch-1} to epoch {epoch}')

            trackoutdir = epochdir / f'from_t{epoch-1}'
            with open(tracking_config,) as f:
                opt_tracking = json.load(f)
            opt_tracking['output_dir'] = trackoutdir
            pairs = [[images[cam0].get_image_path(epoch-1),
                      images[cam0].get_image_path(epoch)],
                     [images[cam1].get_image_path(epoch-1),
                      images[cam1].get_image_path(epoch)]
                     ]
            prevs = [features[cam0][epoch-1].get_features_as_dict(),
                     features[cam1][epoch-1].get_features_as_dict()]
            tracked_cam0, tracked_cam1 = track_matches(
                pairs, maskBB, prevs, opt_tracking)
            # TODO: tenere traccia dell'epoca in cui è stato trovato il match
            # TODO: Check bounding box in tracking
            # TODO: clean tracking code

            # Store all matches in features structure
            features[cam0][epoch].append_features({'kpts': tracked_cam0['keypoints1'],
                                                   'descr': tracked_cam0['descriptors1'],
                                                   'score': tracked_cam0['scores1']})
            features[cam1][epoch].append_features({'kpts': tracked_cam1['keypoints1'],
                                                   'descr': tracked_cam1['descriptors1'],
                                                   'score': tracked_cam1['scores1']})

            # Run Pydegensac to estimate F matrix and reject outliers
            F, inlMask = pydegensac.findFundamentalMatrix(features[cam0][epoch].get_keypoints(),
                                                          features[cam1][epoch].get_keypoints(
            ),
                px_th=2, conf=0.9, max_iters=100000,
                laf_consistensy_coef=-1.0,
                error_type='sampson',
                symmetric_error_check=True,
                enable_degeneracy_check=True,
            )
            F_matrix.append(F)
            print('Matches at epoch {}: pydegensac found {} inliers ({:.2f}%)'.format(epoch, inlMask.sum(),
                                                                                      inlMask.sum()*100 / len(features[cam0][epoch])))

        # Write matched points to disk
        im_stems = images[cam0].get_image_stem(
            epoch), images[cam1].get_image_stem(epoch)
        for jj, cam in enumerate(cam_names):
            features[cam][epoch].save_as_txt(
                epochdir / f'{im_stems[jj]}_mktps.txt')
        with open(epochdir / f'{im_stems[0]}_{im_stems[1]}_features.pickle', 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        last_match_path = create_directory('res/last_epoch')
        with open(last_match_path / 'last_features.pickle', 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Matching completed')

elif not features[cam0]:
    last_match_path = 'res/last_epoch/last_features.pickle'
    with open(last_match_path, 'rb') as f:
        features = pickle.load(f)
        print("Loaded previous matches")
else:
    print("Features already present, nothing was changed.")


## SfM ##
'''
Notes 
'''
# TODO: put parameters, swithches and pre-processing all togheter at the beginning.

# Parameters
target_paths = [Path('data/target_image_p2.txt'),
                Path('data/target_image_p3.txt')]

# Coregistration switches
# TODO: implement these swithces as proprierty of each camera camera Class
# do_coregistration: If True, try to coregister point clouds based on n points
do_coregistration = False    
# fix_both_cameras: if False, estimate EO of cam2 with relative orientation, otherwise keep both cameras fixed.
fix_both_cameras = False

# On-Off switches
do_viz = False
do_SOR_filter = True
rotate_RS = False

# Iizialize variables
cameras = dict.fromkeys(cam_names)
cameras[cam0], cameras[cam1] = [], []
pcd = []
tform = []
img0 = images[cam0][0]
h, w, _ = img0.shape

# Build reference camera objects with only known interior orientation
ref_cams = dict.fromkeys(cam_names)
for jj, cam in enumerate(cam_names):
    ref_cams[cam] = Camera(width=6000, heigth=400,
                           calib_path=Path(calibFld) / f'{cam}.txt')

# Read target image coordinates
targets = Targets(cam_id=[0, 1],  im_coord_path=target_paths)

# Camera baseline
# TODO: put in a json file for general configuration
C1_meta = np.array([416651.5248, 5091109.9121, 1858.9084])   # IMG_2092
C2_meta = np.array([416622.2755, 5091364.5071, 1902.4053])   # IMG_0481
# [m] From Metashape model in July
camWorldBaseline = np.linalg.norm(C1_meta - C2_meta)

#--- SfM ---#
for epoch in epoches_to_process:
    # epoch = 0
    print(f'Reconstructing epoch {epoch}...')

    # Initialize Intrinsics
    # TODO: replace append with insert or a more robust data structure...
    for cam in cam_names:
        cameras[cam].append(Camera(width=ref_cams[cam].width,
                                   heigth=ref_cams[cam].heigth,
                                   K=ref_cams[cam].K,
                                   dist=ref_cams[cam].dist,
                                   ))

    # Estimate Realtive Pose with Essential Matrix
    R, t, valid = estimate_pose(features[cam0][epoch].get_keypoints(),
                                features[cam1][epoch].get_keypoints(),
                                cameras[cam0][epoch].K,
                                cameras[cam1][epoch].K,
                                thresh=1, conf=0.9999,
                                )
    print('Computing relative pose. Valid points: {}/{}'.format(valid.sum(), len(valid)))

    # Update cameras extrinsics
    cameras[cam1][epoch].R = R
    cameras[cam1][epoch].t = t.reshape(3, 1)
    cameras[cam1][epoch].Rt_to_extrinsics()
    cameras[cam1][epoch].extrinsics_to_pose()

    # Scale model by using camera baseline
    camRelOriBaseline = np.linalg.norm(cameras[cam0][epoch].get_C_from_pose() -
                                       cameras[cam1][epoch].get_C_from_pose()
                                       )
    scale_fct = camWorldBaseline / camRelOriBaseline
    T = np.eye(4)
    T[0:3, 0:3] = T[0:3, 0:3] * scale_fct

    # Apply scale to camera extrinsics and update camera proprierties
    cameras[cam1][epoch].pose[:, 3:4] = np.dot(
        T, cameras[cam1][epoch].pose[:, 3:4])
    cameras[cam1][epoch].pose_to_extrinsics()
    cameras[cam1][epoch].update_camera_from_extrinsics()

    # TODO: make wrappers to handle RS transformations
    #   (or use external librariesl)

    if do_coregistration:
        # Triangulate targets
        # TODO: make wrapper around undistort points
        pts0_und = undistort_points(targets.get_im_coord(0)[epoch],
                                    cameras[cam0][epoch]
                                    )
        pts1_und = undistort_points(targets.get_im_coord(1)[epoch],
                                    cameras[cam1][epoch]
                                    )
        M, status = iterative_LS_triangulation(pts0_und, cameras[cam0][epoch].P,
                                               pts1_und, cameras[cam1][epoch].P,
                                               )
        targets.append_obj_cord(M)

        # Estimate rigid body transformation between first epoch RS and current epoch RS
        # TODO: make a Wrapper for this
        v0 = np.concatenate((cameras[cam0][0].C,
                             cameras[cam1][0].C,
                             targets.get_obj_coord()[0].reshape(3, 1),
                             ), axis=1)
        v1 = np.concatenate((cameras[cam0][epoch].C,
                             cameras[cam1][epoch].C,
                             targets.get_obj_coord()[epoch].reshape(3, 1),
                             ), axis=1)
        tform.append(affine_matrix_from_points(
            v1, v0, shear=False, scale=False, usesvd=True))
        print('Point cloud coregistered based on {len(v0)} points.')

    elif epoch > 0:
        # Fix the EO of both the cameras as those estimated in the first epoch
        for cam in cam_names:
            cameras[cam][epoch] = cameras[cam][0]
        print('Camera exterior orientation fixed to that of the master cameras.')

    #--- Triangulate Points ---#
    # TODO: test differences between using LS iterative or linear with SVD
    # TODO: put in a separate function which includes undistortion and then triangulation
    pts0_und = undistort_points(features[cam0][epoch].get_keypoints(),
                                cameras[cam0][epoch]
                                )
    pts1_und = undistort_points(features[cam1][epoch].get_keypoints(),
                                cameras[cam1][epoch]
                                )
    points3d, status = iterative_LS_triangulation(pts0_und, cameras[cam0][epoch].P,
                                                  pts1_und, cameras[cam1][epoch].P,
                                                  )
    print(f'Point triangulation succeded: {status.sum()/status.size}.')

    # Interpolate colors from image
    jj = 1
    image = cv2.cvtColor(images[cam_names[jj]][epoch], cv2.COLOR_BGR2RGB)
    # TODO: include color conversion in function interpolate_point_colors
    points3d_cols = interpolate_point_colors(points3d, image,
                                             cameras[cam_names[jj]][epoch],
                                             )
    print(f'Color interpolated on image {jj} ')

    if do_coregistration:
        # Apply rigid body transformation to triangulated points
        pts = np.dot(tform[epoch], convert_to_homogeneous(points3d.T))
        points3d = convert_from_homogeneous(pts).T

    # Create point cloud and save .ply to disk
    pcd_epc = create_point_cloud(points3d, points3d_cols)

    # Filter outliers in point cloud with SOR filter
    if do_SOR_filter:
        cl, ind = pcd_epc.remove_statistical_outlier(
            nb_neighbors=10, std_ratio=3.0)
        if do_viz:
            display_pc_inliers(pcd_epc, ind)
        pcd_epc = pcd_epc.select_by_index(ind)
        print("Point cloud filtered by Statistical Oulier Removal")

    # Visualize point cloud
    if do_viz:
        cam_syms = []
        cam_colors = [[1, 0, 0], [0, 0, 1]]
        for i, cam in enumerate(cam_names):
            cam_syms.append(make_camera_pyramid(cameras[cam][epoch],
                                                color=cam_colors[i],
                                                focal_len_scaled=30,
                                                ))
        win_name = f'Sparse point cloud - Epoch: {epoch} - Num pts: {len(np.asarray(pcd_epc.points))}'
        o3d.visualization.draw_geometries([pcd_epc,  cam_syms[0], cam_syms[1]], window_name=win_name,
                                          width=1280, height=720,
                                          left=300, top=200)
    # Write point cloud to disk and store it in Point Cloud List
    write_ply(pcd_epc, f'res/pt_clouds/sparse_pts_t{epoch}.ply')
    pcd.append(pcd_epc)

    print('Done.')

# Visualize all points clouds together
o3d.visualization.draw_geometries([pcd[0]],
                                  window_name='All epoches',
                                  width=1280, height=720,
                                  left=300, top=200,
                                  )

fig, ax = plt.subplots(1,2)
fig.tight_layout()
for i, cam in enumerate(cam_names):
    projections = project_points(points3d,
                                 cameras[cam][epoch],
                                 )
    im = cv2.cvtColor(images[cam][epoch], cv2.COLOR_BGR2RGB)
    # im = undistort_image(im, cameras[cam][epoch])
    ax[i].imshow(im)
    ax[i].scatter(projections[:, 0], projections[:, 1],
               s=10, c='r', marker='o',  
               alpha=0.5, edgecolors='k',
               )   
    ax[i].set_title(cam)
    
    
   
# %% DSM
# TODO: implement better DSM class


print('DSM and orthophoto generation started')
res = 0.03
dsms = []
ortofoto = dict.fromkeys(cam_names)
ortofoto[cam0], ortofoto[cam1] = [], []

for epoch in epoches_to_process:
    print(f'Epoch {epoch}')
    dsms.append(build_dsm(np.asarray(pcd[epoch].points),
                          dsm_step=res,
                          make_dsm_plot=False,
                          save_path=f'res/dsm/dsm_app_epoch_{epoch}.tif'
                          ))
    print('DSM built.')
    for cam in cam_names:
        fout_name = f'res/ortofoto/ortofoto_app_cam_{cam}_epc_{epoch}.tif'
        ortofoto[cam].append(generate_ortophoto(cv2.cvtColor(images[cam][epoch], cv2.COLOR_BGR2RGB),
                                                dsms[epoch], cameras[cam][epoch],
                                                save_path=fout_name,
                                                ))
    print('Orthophotos built.')

# fig, ax = plt.subplots()
# ax.imshow(ortofoto[1])


# %% DENSE MATCHING

# Init
epoch = 0
sgm_path = Path('sgm')
downsample = 0.25
fast_viz = True

stem0, stem1 = images[cam0].get_image_stem(
    epoch), images[cam1].get_image_stem(epoch)
img0, img1 = images[cam0][epoch], images[cam1][epoch]
h, w, _ = img0.shape

# pts0, pts1 = features[cam0][epoch].get_keypoints(), features[cam1][epoch].get_keypoints()
# F, inlMask = pydegensac.findFundamentalMatrix(pts0, pts1, px_th=1, conf=0.99999,
#                                               max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson',
#                                               symmetric_error_check=True, enable_degeneracy_check=True)


#--- Rectify calibrated ---#
left_cam = cameras[cam1][epoch]
right_cam = cameras[cam1][epoch]
left_img = images[cam1][epoch]
rigth_img = images[cam0][epoch]


#
rectOut = cv2.stereoRectify(left_cam.K, left_cam.dist,
                            right_cam.K, right_cam.dist,
                            (w, h), left_cam.R, left_cam.t,
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
                                           size=(w, h),
                                           m1type=cv2.CV_32FC1
                                           )
map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix=right_cam.K,
                                           distCoeffs=right_cam.dist,
                                           R=R1,
                                           newCameraMatrix=P1,
                                           size=(w, h),
                                           m1type=cv2.CV_32FC1
                                           )
img0_rect = cv2.remap(left_img, map0x, map0y,
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
img1_rect = cv2.remap(rigth_img, map1x, map1y,
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
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


# %%
#--- Recify Uncalibrated ---#

# Udistort images
img0, img1 = images[cam1][epoch], images[cam0][epoch]
name0 = str(sgm_path / 'und' / (stem0 + "_undistorted.jpg"))
name1 = str(sgm_path / 'und' / (stem1 + "_undistorted.jpg"))
img0, K0_scaled = undistort_image(
    img0, cameras[cam1][epoch].K, cameras[cam1][epoch].dist, downsample, name0)
img1, K1_scaled = undistort_image(
    img1, cameras[cam1][epoch].K, cameras[cam1][epoch].dist, downsample, name1)
h, w, _ = img0.shape

# Undistot points and compute F matrx
pts0 = features[cam0][epoch].get_keypoints()*downsample
pts1 = features[cam1][epoch].get_keypoints()*downsample
pts0 = cv2.undistortPoints(
    pts0, cameras[cam1][epoch].K, cameras[cam1][epoch].dist, None, cameras[cam1][epoch].K)
pts1 = cv2.undistortPoints(
    pts1, cameras[cam1][epoch].K, cameras[cam1][epoch].dist, None, cameras[cam1][epoch].K)
F, inlMask = pydegensac.findFundamentalMatrix(pts0, pts1, px_th=1, conf=0.99999,
                                              max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson',
                                              symmetric_error_check=True, enable_degeneracy_check=True)
print('Pydegensac: {} inliers ({:.2f}%)'.format(
    inlMask.sum(), inlMask.sum()*100 / len(pts0)))

# Rectify uncalibrated
success, H1, H0 = cv2.stereoRectifyUncalibrated(pts0, pts1, F, (w, h))
img0_rectified = cv2.warpPerspective(img0, H0, (w, h))
img1_rectified = cv2.warpPerspective(img1, H1, (w, h))

# write images to disk
path0 = str(sgm_path / 'rectified' / (stem0 + "_rectified.jpg"))
path1 = str(sgm_path / 'rectified' / (stem1 + "_rectified.jpg"))
cv2.imwrite(path0, img0_rectified)
cv2.imwrite(path1, img1_rectified)

if fast_viz:
    print()
else:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img0_rectified, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2RGB))
    plt.show()


# %%


#--- Find epilines corresponding to points in right image (second image) and drawing its lines on left image ---#
img0, img1 = images[cam0][0], images[cam1][0]

lines0 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, F)
lines0 = lines0.reshape(-1, 3)
img0_epiplines, _ = draw_epip_lines(img0, img1, lines0, pts0, pts1)

# Find epilines corresponding to points in left image (first image) and drawing its lines on right image
lines1 = cv2.computeCorrespondEpilines(pts0.reshape(-1, 1, 2), 1, F)
lines1 = lines1.reshape(-1, 3)
img1_epiplines, _ = draw_epip_lines(
    img1, img0, lines1, pts1, pts0, fast_viz=True)

if fast_viz:
    cv2.imwrite(str(sgm_path / (stem0 + "_epiplines.jpg")), img0_epiplines)
    cv2.imwrite(str(sgm_path / (stem1 + "_epiplines.jpg")), img1_epiplines)
else:
    plt.subplot(121), plt.imshow(img0_epiplines)
    plt.subplot(122), plt.imshow(img1_epiplines)
    plt.show()

#--- Draw keypoints and matches ---#
pts0_rect = cv2.perspectiveTransform(
    np.float32(pts0).reshape(-1, 1, 2), H0).reshape(-1, 2)
pts1_rect = cv2.perspectiveTransform(
    np.float32(pts1).reshape(-1, 1, 2), H1).reshape(-1, 2)

# img0_rect_kpts = img0.copy()
img0 = cv2.cvtColor(images[cam0][0], cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(images[cam1][0], cv2.COLOR_BGR2GRAY)
pts0, pts1 = features[cam0]['mkpts0'], features[cam0]['mkpts1']
img0_kpts = cv2.drawKeypoints(img0, cv2.KeyPoint.convert(
    pts0), img0, (), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img1_kpts = cv2.drawKeypoints(img1, cv2.KeyPoint.convert(
    pts1), img1, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('kpts0.jpg', img0_kpts)
cv2.imwrite('kpts1.jpg', img1_kpts)

make_matching_plot(img0, img1, pts0, pts1, path='matches.jpg')
