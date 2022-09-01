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

# %%

from lib.validate_inputs import validate
import numpy as np
import cv2
import pickle
import json
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import pydegensac

from pathlib import Path
from easydict import EasyDict as edict

from lib.config import parse_yaml_cfg
from lib.classes import (Camera, Imageds, Features, Targets)
from lib.sfm.two_view_geometry import Two_view_geometry
from lib.sfm.triangulation import Triangulate
from lib.match_pairs import match_pair
from lib.track_matches import track_matches

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

from thirdparty.transformations import affine_matrix_from_points

# Parse options from yaml file
cfg_file = 'config/config_base.yaml'
cfg = parse_yaml_cfg(cfg_file)

# Inizialize Variables
cams = cfg.paths.cam_names
features = dict.fromkeys(cams)  # @TODO: put this in an inizialization function

# Create Image Datastore objects
images = dict.fromkeys(cams)  # @TODO: put this in an inizialization function
for cam in cams:
    images[cam] = Imageds(cfg.paths.imdir / cam)

cfg = validate(cfg, images)

''' Perform matching and tracking '''
# Load matching and tracking configurations
with open(cfg.matching_cfg) as f:
    opt_matching = edict(json.load(f))
with open(cfg.tracking_cfg) as f:
    opt_tracking = edict(json.load(f))

epoch = 0
if cfg.proc.do_matching:
    for cam in cams:
        features[cam] = []

    for epoch in cfg.proc.epoch_to_process:
        print(f'Processing epoch {epoch}...')

        # opt_matching = cfg.matching.copy()
        epochdir = Path(cfg.paths.resdir) / f'epoch_{epoch}'

        #-- Find Matches at current epoch --#
        print(f'Run Superglue to find matches at epoch {epoch}')
        opt_matching.output_dir = epochdir
        pair = [
            images[cams[0]].get_image_path(epoch),
            images[cams[1]].get_image_path(epoch)
        ]
        # Call matching function
        matchedPts, matchedDescriptors, matchedPtsScores = match_pair(
            pair, cfg.images.bbox, opt_matching
        )

        # Store matches in features structure
        for jj, cam in enumerate(cams):
            # Dict keys are the cameras names, internal list contain epoches
            features[cam].append(Features())
            features[cam][epoch].append_features({
                'kpts': matchedPts[jj],
                'descr': matchedDescriptors[jj],
                'score': matchedPtsScores[jj]
            })
            # @TODO: Store match confidence!

        #=== Track previous matches at current epoch ===#
        if cfg.proc.do_tracking and epoch > 0:
            print(f'Track points from epoch {epoch-1} to epoch {epoch}')

            trackoutdir = epochdir / f'from_t{epoch-1}'
            opt_tracking['output_dir'] = trackoutdir
            pairs = [
                [images[cams[0]].get_image_path(epoch-1),
                    images[cams[0]].get_image_path(epoch)],
                [images[cams[1]].get_image_path(epoch-1),
                    images[cams[1]].get_image_path(epoch)],
            ]
            prevs = [
                features[cams[0]][epoch-1].get_features_as_dict(),
                features[cams[1]][epoch-1].get_features_as_dict()
            ]
            # Call actual tracking function
            tracked_cam0, tracked_cam1 = track_matches(
                pairs, cfg.images.bbox, prevs, opt_tracking)
            # @TODO: keep track of the epoch in which feature is matched
            # @TODO: Check bounding box in tracking
            # @TODO: clean tracking code

            # Store all matches in features structure
            features[cams[0]][epoch].append_features(tracked_cam0)
            features[cams[1]][epoch].append_features(tracked_cam1)

        # Run Pydegensac to estimate F matrix and reject outliers
        F, inlMask = pydegensac.findFundamentalMatrix(
            features[cams[0]][epoch].get_keypoints(),
            features[cams[1]][epoch].get_keypoints(),
            px_th=1.5, conf=0.99999, max_iters=10000,
            laf_consistensy_coef=-1.0,
            error_type='sampson',
            symmetric_error_check=True,
            enable_degeneracy_check=True,
        )
        print(f'Matching at epoch {epoch}: pydegensac found {inlMask.sum()} \
            inliers ({inlMask.sum()*100/len(features[cams[0]][epoch]):.2f}%)')
        features[cams[0]][epoch].remove_outliers_features(inlMask)
        features[cams[1]][epoch].remove_outliers_features(inlMask)

        # Write matched points to disk
        im_stems = images[cams[0]].get_image_stem(
            epoch), images[cams[1]].get_image_stem(epoch)
        for jj, cam in enumerate(cams):
            features[cam][epoch].save_as_txt(
                epochdir / f'{im_stems[jj]}_mktps.txt')
        with open(epochdir / f'{im_stems[0]}_{im_stems[1]}_features.pickle', 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        last_match_path = create_directory('res/last_epoch')
        with open(last_match_path / 'last_features.pickle', 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

        print('Matching completed')

elif not features[cams[0]]:
    last_match_path = 'res/last_epoch/last_features.pickle'
    with open(last_match_path, 'rb') as f:
        features = pickle.load(f)
        print("Loaded previous matches")
else:
    print("Features already present, nothing was changed.")

# %%
''' SfM '''

# Initialize variables
cameras = dict.fromkeys(cams)
cameras[cams[0]], cameras[cams[1]] = [], []
pcd = []
tform = []
h, w = 4000, 6000

# Build reference camera objects with only known interior orientation
ref_cams = dict.fromkeys(cams)
for jj, cam in enumerate(cams):
    ref_cams[cam] = Camera(
        width=6000, height=400,
        calib_path=cfg.paths.caldir / f'{cam}.txt'
    )

# Read target image coordinates
targets = Targets(cam_id=[0, 1],  im_coord_path=cfg.georef.target_paths)

# Camera baseline
baseline_world = np.linalg.norm(
    cfg.georef.camera_centers_world[0] - cfg.georef.camera_centers_world[1]
)

for epoch in cfg.proc.epoch_to_process:
    # epoch = 0
    print(f'Reconstructing epoch {epoch}...')

    # Initialize Intrinsics
    ''' Inizialize Camera Intrinsics at every epoch setting them equal to
        the reference cameras ones.
    '''
    # @TODO: replace append with insert or a more robust data structure...
    for cam in cams:
        cameras[cam].append(
            Camera(
                width=ref_cams[cam].width,
                height=ref_cams[cam].height,
                K=ref_cams[cam].K,
                dist=ref_cams[cam].dist,
            )
        )

    # Perform Relative orientation of the two cameras
    ''' Initialize Two_view_geometry class with a list containing the two cameras
        and a list contaning the matched features location on each camera.
    '''
    relative_ori = Two_view_geometry(
        [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
        [features[cams[0]][epoch].get_keypoints(),
            features[cams[1]][epoch].get_keypoints()],
    )
    relative_ori.relative_orientation(threshold=1.5, confidence=0.999999)
    relative_ori.scale_model_with_baseline(baseline_world)

    # Save relative orientation results in Camera objects of current epoch
    cameras[cams[0]][epoch] = relative_ori.cameras[0]
    cameras[cams[1]][epoch] = relative_ori.cameras[1]

    if cfg.proc.do_coregistration:
        # @TODO: make wrappers to handle RS transformations

        # Triangulate targets
        triangulate = Triangulate(
            [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
            [targets.get_im_coord(0)[epoch],
                targets.get_im_coord(1)[epoch]],
        )
        targets.append_obj_cord(triangulate.triangulate_two_views())

        # Estimate rigid body transformation between first epoch RS and current epoch RS
        # @TODO: make a wrapper for this
        v0 = np.concatenate((cameras[cams[0]][0].C,
                             cameras[cams[1]][0].C,
                             targets.get_obj_coord()[0].reshape(3, 1),
                             ), axis=1)
        v1 = np.concatenate((cameras[cams[0]][epoch].C,
                             cameras[cams[1]][epoch].C,
                             targets.get_obj_coord()[epoch].reshape(3, 1),
                             ), axis=1)
        tform.append(affine_matrix_from_points(
            v1, v0, shear=False, scale=False, usesvd=True))
        print('Point cloud coregistered based on {len(v0)} points.')

    elif epoch > 0:
        # Fix the EO of both the cameras as those estimated in the first epoch
        for cam in cams:
            cameras[cam][epoch] = cameras[cam][0]
        print('Camera exterior orientation fixed to that of the master cameras.')

    #--- Triangulate Points ---#
    ''' Initialize Triangulate class with a list containing the two cameras
        and a list contaning the matched features location on each camera.
        Triangulated points are saved as points3d proprierty of the
        Triangulate object (eg., triangulation.points3d)
    '''
    triangulation = Triangulate(
        [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
        [
            features[cams[0]][epoch].get_keypoints(),
            features[cams[1]][epoch].get_keypoints()
        ]
    )
    triangulation.triangulate_two_views()
    triangulation.interpolate_colors_from_image(
        images[cams[1]][epoch],
        cameras[cams[1]][epoch],
        convert_BRG2RGB=True,
    )

    if cfg.proc.do_coregistration:
        # Apply rigid body transformation to triangulated points
        # @TODO: make wrapper for apply transformation to arrays
        pts = np.dot(tform[epoch],
                     convert_to_homogeneous(triangulation.points3d.T)
                     )
        triangulation.points3d = convert_from_homogeneous(pts).T

    # Create point cloud and save .ply to disk
    pcd_epc = create_point_cloud(
        triangulation.points3d, triangulation.colors)

    # Filter outliers in point cloud with SOR filter
    if cfg.other.do_SOR_filter:
        _, ind = pcd_epc.remove_statistical_outlier(nb_neighbors=10,
                                                    std_ratio=3.0)
        #     display_pc_inliers(pcd_epc, ind)
        pcd_epc = pcd_epc.select_by_index(ind)
        print("Point cloud filtered by Statistical Oulier Removal")

    # Write point cloud to disk and store it in Point Cloud List
    write_ply(pcd_epc, f'res/pt_clouds/sparse_pts_t{epoch}.ply')
    pcd.append(pcd_epc)

print('Done.')

# %%
''' Some various visualization functions'''

# Visualize point cloud at epoch x
epoch = 0
display_point_cloud(
    pcd[epoch],
    [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
    win_name=f'Point cloud at Epoch {epoch} - num points: {len(pcd[epoch].points)}'
)

# Plot detected features on stereo pair
fig, axes = plt.subplots(2, 1)
for i, cam in enumerate(cams):
    plot_features(
        images[cam][epoch],
        features[cam][epoch].get_keypoints(),
        cam, axes[i],
    )

# Plot projections of 3d points on stereo-pair (after triangulation)
triangulation = Triangulate(
    [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
    [
        features[cams[0]][epoch].get_keypoints(),
        features[cams[1]][epoch].get_keypoints()
    ]
)
triangulation.triangulate_two_views()
fig, axes = plt.subplots(2, 1)
for i, cam in enumerate(cams):
    plot_projections(triangulation.points3d, cameras[cam][epoch],
                     images[cam][epoch], cam, axes[i]
                     )

# Plot reprojection error (note, values are normalized so far...)
cam = cams[0]
triangulation = Triangulate(
    [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
    [
        features[cams[0]][epoch].get_keypoints(),
        features[cams[1]][epoch].get_keypoints()
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
# features[cams[0]][epoch].remove_outliers_features(inliers)
# features[cams[1]][epoch].remove_outliers_features(inliers)
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
''' Compute DSM and orthophotos '''
# @TODO: implement better DSM class

print('DSM and orthophoto generation started')
res = 0.03
xlim = [-100., 80.]
ylim = [-10., 65.]

dsms = []
ortofoto = dict.fromkeys(cams)
ortofoto[cams[0]], ortofoto[cams[1]] = [], []
for epoch in cfg.proc.epoch_to_process:
    print(f'Epoch {epoch}')
    dsms.append(build_dsm(np.asarray(pcd[epoch].points),
                          dsm_step=res,
                          xlim=xlim, ylim=ylim,
                          make_dsm_plot=False,
                          # fill_value = ,
                          save_path=f'res/dsm/dsm_app_epoch_{epoch}.tif'
                          ))
    print('DSM built.')
    for cam in cams:
        fout_name = f'res/ortofoto/ortofoto_app_cam_{cam}_epc_{epoch}.tif'
        ortofoto[cam].append(generate_ortophoto(cv2.cvtColor(images[cam][epoch], cv2.COLOR_BGR2RGB),
                                                dsms[epoch], cameras[cam][epoch],
                                                xlim=xlim, ylim=ylim,
                                                save_path=fout_name,
                                                ))
    print('Orthophotos built.')


# if __name__ == '__main__':
#     main()
