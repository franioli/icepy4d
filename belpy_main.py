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

import matplotlib.cm as cm
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
from lib.sfm.triangulation import Triangulate
from lib.sfm.two_view_geometry import Two_view_geometry

from lib.geometry import (project_points, 
                          compute_reprojection_error
                          )
from lib.utils import (build_dsm,
                       generate_ortophoto,
                       )
from lib.point_clouds import (create_point_cloud,
                              write_ply,
                              )
from lib.visualization import (display_point_cloud,
                               display_pc_inliers,
                               plot_features,
                               plot_projections,
                               )
from lib.misc import (convert_to_homogeneous,
                      convert_from_homogeneous,
                      create_directory,
                      )
from lib.thirdParts.transformations import affine_matrix_from_points


def main() -> None:
    #---  Parameters  ---#
    # TODO: put parameters in parser or in json file

    # - Folders and paths
    imFld = 'data/img'
    calibFld = 'data/calib'
    matching_config = 'config/opt_matching.json'
    tracking_config = 'config/opt_tracking.json'
    res_folder = 'res'

    # - CAMERAS
    cam_names = ['p0', 'p1']  # ['p0_00', 'p1_00'] #
    ''' Note that the calibration file must have the same name as the camera.'''

    # - Targets
    target_paths = [
        Path('data/target_image_p0.txt'),
        Path('data/target_image_p1.txt'),
        # Path('data/target_image_p1.txt'),
        # Path('data/target_image_p0.txt'),
    ]

    # Find new matches
    find_matches = False

    # Epoches to process
    # It can be 'all' for processing all the epochs or a list with the epoches to be processed
    epoches_to_process = 'all'  # [x for x in range(15)]  # [0] #

    # Coregistration switches
    # TODO: implement these swithces as proprierty of each camera camera Class
    # do_coregistration: If True, try to coregister point clouds based on n points (still have to fully implement it)
    do_coregistration = False

    # fix_both_cameras: if False, estimate EO of cam2 with relative orientation, otherwise keep both cameras fixed.
    # fix_both_cameras = False

    # Other On-Off switches
    do_viz = False
    do_SOR_filter = True
    rotate_RS = False

    # - Bounding box for processing the images from the two cameras
    maskBB = [[400, 1900, 5500, 3500], [300, 1800, 5700, 3500]]

    # Camera centers obtained from Metashape model in July [m] 
    camera_centers_world = np.array([
            [416651.5248, 5091109.9121, 1858.9084],   # IMG_2092
            [416622.2755, 5091364.5071, 1902.4053],   # IMG_0481
        ])
  
    ''' Perform matching and tracking '''
    
    #  Inizialize Variables
    # TODO: replace all lists with dictionaries
    # TODO: replace cam0, cam1 with iterable objects
    images = dict.fromkeys(cam_names)
    features = dict.fromkeys(cam_names)
    f_matrixes = []

    numCams = len(cam_names)
    cam0, cam1 = cam_names[0], cam_names[1]

    # - Create Image Datastore objects
    for cam in cam_names:
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
                Dict keys are the cameras names, internal list are the epoches...
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
                # TODO: keep track of the epoch in which feature is matched
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
            F, inlMask = pydegensac.findFundamentalMatrix(
                features[cam0][epoch].get_keypoints(),
                features[cam1][epoch].get_keypoints(),
                px_th=1.5, conf=0.99999, max_iters=10000,
                laf_consistensy_coef=-1.0,
                error_type='sampson',
                symmetric_error_check=True,
                enable_degeneracy_check=True,
                )
            f_matrixes.append(F)
            print(f'Matching at epoch {epoch}: pydegensac found {inlMask.sum()} \
                inliers ({inlMask.sum()*100/len(features[cam0][epoch]):.2f}%)')
            features[cam0][epoch].remove_outliers_features(inlMask)
            features[cam1][epoch].remove_outliers_features(inlMask)

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

    # %%
    ''' SfM '''
    
    # Initialize variables
    cameras = dict.fromkeys(cam_names)
    cameras[cam0], cameras[cam1] = [], []
    pcd = []
    tform = []
    h, w = 4000, 6000

    # Build reference camera objects with only known interior orientation
    ref_cams = dict.fromkeys(cam_names)
    for jj, cam in enumerate(cam_names):
        ref_cams[cam] = Camera(
            width=6000, height=400,
            calib_path=Path(calibFld) / f'{cam}.txt'
            )

    # Read target image coordinates
    targets = Targets(cam_id=[0, 1],  im_coord_path=target_paths)

    # Camera baseline 
    baseline_world = np.linalg.norm(
        camera_centers_world[0] - camera_centers_world[1]
    )

    for epoch in epoches_to_process:
        # epoch = 0
        print(f'Reconstructing epoch {epoch}...')

        # Initialize Intrinsics
        # TODO: replace append with insert or a more robust data structure...
        for cam in cam_names:
            cameras[cam].append(
                Camera(
                    width=ref_cams[cam].width,
                    height=ref_cams[cam].height,
                    K=ref_cams[cam].K,
                    dist=ref_cams[cam].dist,
                )
        )

        # Perform Relative orientation of the two cameras
        relative_ori = Two_view_geometry(
            [cameras[cam0][epoch], cameras[cam1][epoch]],
            [features[cam0][epoch].get_keypoints(), features[cam1]
            [epoch].get_keypoints()],
        )
        relative_ori.relative_orientation(threshold=1.5, confidence=0.999999)
        relative_ori.scale_model_with_baseline(baseline_world)

        # Save relative orientation results in Camera objects of current epoch
        cameras[cam0][epoch] = relative_ori.cameras[0]
        cameras[cam1][epoch] = relative_ori.cameras[1]

        # TODO: make wrappers to handle RS transformations
        #   (or use external libraries)
        if do_coregistration:
            # Triangulate targets
            triangulate = Triangulate(
                [cameras[cam0][epoch], cameras[cam1][epoch]],
                [targets.get_im_coord(0)[epoch],
                                      targets.get_im_coord(1)[epoch]],
        )
            targets.append_obj_cord(triangulate.triangulate_two_views())

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
        triangulation = Triangulate(
            [cameras[cam0][epoch], cameras[cam1][epoch]],
            [
                features[cam0][epoch].get_keypoints(),
                features[cam1][epoch].get_keypoints()
             ]
        )
        triangulation.triangulate_two_views()
        triangulation.interpolate_colors_from_image(
            images[cam1][epoch],
            cameras[cam1][epoch],
            convert_BRG2RGB=True,
        )

        if do_coregistration:
            # Apply rigid body transformation to triangulated points
            pts = np.dot(tform[epoch],
                        convert_to_homogeneous(triangulation.points3d.T)
                        )
            triangulation.points3d = convert_from_homogeneous(pts).T

        # Create point cloud and save .ply to disk
        pcd_epc = create_point_cloud(
            triangulation.points3d, triangulation.colors)

        # Filter outliers in point cloud with SOR filter
        if do_SOR_filter:
            _, ind = pcd_epc.remove_statistical_outlier(nb_neighbors=10,
                                                    std_ratio=3.0)
            if do_viz:
                display_pc_inliers(pcd_epc, ind)
            pcd_epc = pcd_epc.select_by_index(ind)
            print("Point cloud filtered by Statistical Oulier Removal")

        # Write point cloud to disk and store it in Point Cloud List
        write_ply(pcd_epc, f'res/pt_clouds/sparse_pts_t{epoch}.ply')
        pcd.append(pcd_epc)

    print('Done.')

    # %% Some visualization
    ''' Visualization '''
    
    # Visualize point cloud at epoch x
    epoch = 0
    display_point_cloud(pcd[epoch],
                        [
                            cameras[cam0][epoch],
                            cameras[cam1][epoch]],
                        )

    # Plot detected features on stereo pair    
    fig, axes = plt.subplots(2, 1)
    for i, cam in enumerate(cam_names):
        plot_features(
            images[cam][epoch],
            features[cam][epoch].get_keypoints(),
            cam, axes[i],
            )

    # Plot projections of 3d points on stereo-pair (after triangulation)
    triangulation = Triangulate(
        [cameras[cam0][epoch], cameras[cam1][epoch]],
        [
            features[cam0][epoch].get_keypoints(),
            features[cam1][epoch].get_keypoints()
            ]
    )
    triangulation.triangulate_two_views()
    fig, axes = plt.subplots(2, 1)
    for i, cam in enumerate(cam_names):
        plot_projections(triangulation.points3d, cameras[cam][epoch],
                        images[cam][epoch], cam, axes[i]
                        )

    # Plot reprojection error (note, it is normalized so far...)
    cam = cam0
    triangulation = Triangulate(
            [cameras[cam0][epoch], cameras[cam1][epoch]],
            [
                features[cam0][epoch].get_keypoints(),
                features[cam1][epoch].get_keypoints()
            ]
        )
    projections = project_points(
        triangulation.triangulate_two_views(),
        cameras[cam][epoch],
        )
    reprojection_error, rmse = compute_reprojection_error(
        features[cam][epoch].get_keypoints(), 
        projections
        )
    print(f"Reprojection rmse: {rmse}")

    # Reject features with reprojection error magnitude larger than proj_err_max
    # err = reprojection_error[:, 2]
    # proj_err_max = 4
    # inliers = err < proj_err_max
    # print(
    #     f'Rejected points: {np.invert(inliers).sum()}/{len(err)} --> number of inliers matches: {inliers.sum()}')
    # features[cam0][epoch].remove_outliers_features(inliers)
    # features[cam1][epoch].remove_outliers_features(inliers)
    # ''' Now MANUALLY perform again orientation and reprojection error computation... and iterate'''

    # im = cv2.cvtColor(images[cam][epoch], cv2.COLOR_BGR2RGB)
    # viridis = cm.get_cmap('viridis', 8)
    # norm = Colors.Normalize(vmin=err.min(), vmax=err.max())
    # cmap = viridis(norm(err))

    # fig, ax = plt.subplots()
    # fig.tight_layout()
    # ax.imshow(im)
    # scatter = ax.scatter(projections[:, 0], projections[:, 1],
    #                      s=10, c=cmap, marker='o',
    #                      # alpha=0.5, edgecolors='k',
    #                      )
    # ax.set_title(cam)
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label("Reprojection error in y")


    # %% 
    ''' DSM and orthophoto '''
    # TODO: implement better DSM class

    print('DSM and orthophoto generation started')
    res = 0.03
    xlim = [-100., 80.]
    ylim = [-10., 65.]

    dsms = []
    ortofoto = dict.fromkeys(cam_names)
    ortofoto[cam0], ortofoto[cam1] = [], []
    for epoch in epoches_to_process:
        print(f'Epoch {epoch}')
        dsms.append(build_dsm(np.asarray(pcd[epoch].points),
                            dsm_step=res,
                            xlim=xlim, ylim=ylim,
                            make_dsm_plot=False,
                            # fill_value = ,
                            save_path=f'res/dsm/dsm_app_epoch_{epoch}.tif'
                            ))
        print('DSM built.')
        for cam in cam_names:
            fout_name = f'res/ortofoto/ortofoto_app_cam_{cam}_epc_{epoch}.tif'
            ortofoto[cam].append(generate_ortophoto(cv2.cvtColor(images[cam][epoch], cv2.COLOR_BGR2RGB),
                                                    dsms[epoch], cameras[cam][epoch],
                                                    xlim=xlim, ylim=ylim,
                                                    save_path=fout_name,
                                                    ))
        print('Orthophotos built.')


if __name__ == '__main__': 
    main()
    