# %%
import numpy as np
import cv2
import pickle
import json
import matplotlib.pyplot as plt
import pydegensac
import open3d as o3d

from pathlib import Path
from easydict import EasyDict as edict

from lib.classes import (Camera, Imageds, Features, Targets)
from lib.sfm.two_view_geometry import Two_view_geometry
from lib.sfm.absolute_orientation import (Absolute_orientation, 
                                          Space_resection,
                                          )
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
from lib.misc import create_directory
from lib.config import parse_yaml_cfg, validate_inputs

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

cfg = validate_inputs(cfg, images)

''' Perform matching and tracking '''
# Load matching and tracking configurations
with open(cfg.matching_cfg) as f:
    opt_matching = edict(json.load(f))
with open(cfg.tracking_cfg) as f:
    opt_tracking = edict(json.load(f))

# epoch = 0
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
        print('Loaded previous matches')
else:
    print('Features already present, nothing was changed.')


# %%
''' SfM '''


# Initialize variables @TODO: build function for variable inizialization
cameras = dict.fromkeys(cams)
cameras[cams[0]], cameras[cams[1]] = [], []
point_clouds = []
tform = []
im_height, im_width = 4000, 6000
# @TODO: store this information in exif inside an Image Class

# Read target image coordinates and object coordinates 
targets = []
for epoch in cfg.proc.epoch_to_process:
    
    p1_path = cfg.georef.target_dir / (
        images[cams[0]].get_image_stem(epoch)+cfg.georef.target_file_ext
        )
                        
    p2_path = cfg.georef.target_dir / (
        images[cams[1]].get_image_stem(epoch)+cfg.georef.target_file_ext
        )

    targets.append(Targets(
        im_file_path=[p1_path, p2_path],
        obj_file_path='data/target_world_p1.csv'
        )
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
                width=im_width,
                height=im_height,
                calib_path=cfg.paths.caldir / f'{cam}.txt'
            )
        )
            
    #--- At the first epoch, perform Space resection of the first camera by using GCPs. At all other epoches, set camera 1 EO equal to first one. ---#
    # if epoch == 0: 
        # ''' Initialize Single_camera_geometry class with a cameras object'''
        # targets_to_use = ['T2','T3','T4','F2' ]
        # space_resection = Space_resection(cameras[cams[0]][epoch])
        # space_resection.estimate(
        #     targets[epoch].extract_image_coor_by_label(targets_to_use,cam_id=0),
        #     targets[epoch].extract_object_coor_by_label(targets_to_use)
        #     )
        # # Store result in camera 0 object
        # cameras[cams[0]][epoch] = space_resection.camera
    # else:
    #     cameras[cams[0]][epoch] = cameras[cams[0]][0]
    
    #--- Perform Relative orientation of the two cameras ---#
    ''' Initialize Two_view_geometry class with a list containing the two cameras and a list contaning the matched features location on each camera.
    '''
    relative_ori = Two_view_geometry(
        [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
        [features[cams[0]][epoch].get_keypoints(),
         features[cams[1]][epoch].get_keypoints()],
    )
    relative_ori.relative_orientation(
        threshold=1.5, 
        confidence=0.999999, 
        scale_factor=261.606245022935 # 272.888187  #  baseline_world24
        )
    # Store result in camera 1 object
    cameras[cams[1]][epoch] = relative_ori.cameras[1]

    #--- Triangulate Points ---#
    ''' Initialize Triangulate class with a list containing the two cameras
        and a list contaning the matched features location on each camera.
        Triangulated points are saved as points3d proprierty of the
        Triangulate object (eg., triangulation.points3d)
    '''
    triangulation = Triangulate(
        [cameras[cams[0]][epoch], 
         cameras[cams[1]][epoch]],
        [features[cams[0]][epoch].get_keypoints(),
         features[cams[1]][epoch].get_keypoints()]
    )
    points3d = triangulation.triangulate_two_views()
    triangulation.interpolate_colors_from_image(
        images[cams[1]][epoch],
        cameras[cams[1]][epoch],
        convert_BRG2RGB=True,
    )
    
    # # Absolute orientation (-> coregistration on stable points)
    # targets_to_use = ['T2', 'F2'] # 'T4',
    # abs_ori = Absolute_orientation(
    #     (cameras[cams[0]][epoch], cameras[cams[1]][epoch]),
    #     points3d_world=targets[epoch].extract_object_coor_by_label(targets_to_use),
    #     image_points=(
    #         targets[epoch].extract_image_coor_by_label(targets_to_use, cam_id=0),
    #         targets[epoch].extract_image_coor_by_label(targets_to_use, cam_id=1),
    #     )
    # )
    # T = abs_ori.estimate_transformation(
    #     estimate_scale=True,
    #     add_camera_centers=True,
    #     camera_centers_world=tuple(cfg.georef.camera_centers_world)
    # )
                                                
    # points3d = abs_ori.apply_transformation(points3d=points3d)
    
    # Create point cloud and save .ply to disk
    pcd_epc = create_point_cloud(
        points3d, triangulation.colors)

    # Filter outliers in point cloud with SOR filter
    # if cfg.other.do_SOR_filter:
    #     _, ind = pcd_epc.remove_statistical_outlier(nb_neighbors=10,
    #                                                 std_ratio=3.0)
    #     #     display_pc_inliers(pcd_epc, ind)
    #     pcd_epc = pcd_epc.select_by_index(ind)
    #     print('Point cloud filtered by Statistical Oulier Removal')


    # Write point cloud to disk and store it in Point Cloud List
    write_ply(pcd_epc, f'res/pt_clouds/sparse_pts_t{epoch}.ply')
    point_clouds.append(pcd_epc)

print('Done.')


# Visualize point cloud
display_point_cloud(
    point_clouds,
    [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
    plot_scale=10,
)

# %%
# Absolute orientation (lmfit)
from lmfit import Minimizer, minimize, Parameters, fit_report

from lib.least_squares.rototra3d import compute_residuals
from lib.least_squares.utils import print_results
from lib.least_squares.rototra3d import (
    compute_tform_matrix_from_params,
    apply_transformation_to_points
    )

epoch = 0

targets_to_use = ['F2'] # ['T2', 'F2'] # 'T4',
triangulation = Triangulate(
    [
        cameras[cams[0]][epoch], 
        cameras[cams[1]][epoch],
    ],
    [
        targets[epoch].extract_image_coor_by_label(targets_to_use, cam_id=0),
        targets[epoch].extract_image_coor_by_label(targets_to_use, cam_id=1)
    ]
)
triangulation.triangulate_two_views()

# Build arrays
v0 = triangulation.points3d
for cam in cams:
    c =  cameras[cam][epoch].C.reshape(1,3)
    v0 = np.concatenate((v0, c), axis=0)
print(f'V0: {v0}')

v1 = targets[epoch].extract_object_coor_by_label(targets_to_use)
v1 = np.concatenate((v1, cfg.georef.camera_centers_world), axis=0)
print(f'V1: {v1}')

# Initial values
t_ini = np.array(
    [1.46882746e+02, 8.74147624e+01, 9.04722323e+01], 
    dtype='float64'
)
rot_ini = np.array((-1.455234490428092, 0.06619166269889347,
                   0.9470055218154193), 'float64')
m_ini = float(0.0)

# Define Parameters to be optimized
params = Parameters()
params.add('rx', value=rot_ini[0], vary=True)
params.add('ry', value=rot_ini[1], vary=True)
params.add('rz', value=rot_ini[2], vary=True)
params.add('tx', value=t_ini[0], vary=True)
params.add('ty', value=t_ini[1], vary=True)
params.add('tz', value=t_ini[2], vary=True)
params.add('m',  value=m_ini, vary=True)

uncertainty = np.ones(v0.shape)  # Default assigned uncertainty[m]
# uncertainty[0,:] *= 1  # weights for T2
uncertainty[0,:] *= 0.05  # weights for F2
uncertainty[1,:] *= 0.0001  # weights for camera 1
uncertainty[2,:] *= 0.2  # weights for camera 1

# Run Optimization!
weights = 1. / uncertainty
minimizer = Minimizer(
    compute_residuals,
    params,
    fcn_args=(
        v0,
        v1,
    ),
    fcn_kws={
        'weights': weights,
    },
    scale_covar=True,
)
ls_result = minimizer.minimize(method='leastsq')
# fit_report(result)

# Print result
print_results(ls_result, weights)


# Apply transformation to point cloud 
points3d = apply_transformation_to_points(
    points3d = np.asarray(point_clouds[0].points),
    tform = compute_tform_matrix_from_params(ls_result.params)
)
pt_cloud_world = create_point_cloud(
        points3d, triangulation.colors)
write_ply(pt_cloud_world, f'res/pt_clouds/pts_ls_abs_ori.ply')


# %%
''' Export observations for external BBA '''

def export_keypoints(
    filename: str,
    features: Features,
    imageds: Imageds,
    epoch: int = None,
) -> None:
    if epoch is not None:

        cams = list(imageds.keys())

        # Write header to file
        file = open(filename, "w")
        file.write("image_name, feature_id, x, y\n")

        for cam in cams:
            image_name = imageds[cam].get_image_name(epoch)

            # Write image name line
            # NB: must be manually modified if it contains characters of symbols
            file.write(f"{image_name}\n")

            for id, kpt in enumerate(features[cam][epoch].get_keypoints()):
                x, y = kpt
                file.write(
                        f"{id},{x},{y} \n"
                        )

        file.close()
        print("Marker exported successfully")
    else:
        print('please, provide the epoch number.')
        return


def export_points3D(
    filename: str,
    points3D: np.ndarray,
) -> None:
    # Write header to file
    file = open(filename, "w")
    file.write("point_id, X, Y, Z\n")

    for id, pt in enumerate(points3D):
        file.write(f"{id},{pt[0]},{pt[1]},{pt[2]}\n")

    file.close()
    print("Points exported successfully")

epoch = 0

for cam in cams:
    print(f'Cam {cam}')
    print(f'C:\n{cameras[cam][epoch].C}')
    print(f'R:\n{cameras[cam][epoch].R}')

export_keypoints(
    'for_bba/keypoints_280722_for_bba.txt',
    features=features,
    imageds=images,
    epoch=epoch,
)
export_points3D(
    'for_bba/points3d_280722_for_bba.txt',
    points3D=np.asarray(point_clouds[epoch].points)
)

# Targets
targets[epoch].im_coor[0].to_csv('for_bba/targets_p1.txt', index=False)
targets[epoch].im_coor[1].to_csv('for_bba/targets_p2.txt', index=False)
targets[epoch].obj_coor.to_csv('for_bba/targets_world.txt', index=False)

# %%
''' For CALGE'''

# CAMERA EXTERIOR ORIENTATION
from thirdparty.transformations import euler_from_matrix

print(cameras[cams[0]][0].get_C_from_pose() )
print(cameras[cams[1]][0].get_C_from_pose() )
print(np.array(euler_from_matrix(cameras['p1'][0].R)) * 200/np.pi)
print(np.array(euler_from_matrix(cameras['p2'][0].R)) * 200/np.pi)


baseline_world = np.linalg.norm(
    cfg.georef.camera_centers_world[0] - cfg.georef.camera_centers_world[1]
)

print(baseline_world)


# SAVE HOMOLOGOUS POINTS
# NB: Remember to disable SOR filter when computing 3d coordinates of TPs
from lib.io import export_keypoints_for_calge, export_points3D_for_calge

from thirdparty.transformations import euler_from_matrix

epoch = 0
export_keypoints_for_calge('simulaCalge/keypoints_280722.txt',
                           features=features,
                           imageds=images,
                           epoch=epoch,
                           pixel_size_micron=3.773
                           )
export_points3D_for_calge('simulaCalge/points3D_280722.txt',
                           points3D=np.asarray(point_clouds[epoch].points)
                           )

print(cameras['p1'][0].C)
print(cameras['p2'][0].C)


print(np.array(euler_from_matrix(cameras['p1'][0].R)) * 200/np.pi)
print(np.array(euler_from_matrix(cameras['p2'][0].R)) * 200/np.pi)


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
    dsms.append(build_dsm(np.asarray(point_clouds[epoch].points),
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


