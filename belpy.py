import numpy as np
import cv2
import pickle
import json
import pydegensac

from pathlib import Path
from easydict import EasyDict as edict
from copy import deepcopy

from lib.classes import Camera, Imageds, Features, Targets
from lib.sfm.two_view_geometry import Two_view_geometry
from lib.sfm.absolute_orientation import (
    Absolute_orientation,
    Space_resection,
)
from lib.sfm.triangulation import Triangulate
from lib.match_pairs import match_pair
from lib.track_matches import track_matches

from lib.utils import (
    build_dsm,
    generate_ortophoto,
)
from lib.point_clouds import (
    create_point_cloud,
    write_ply,
)
from lib.visualization import display_point_cloud
from lib.misc import create_directory
from lib.config import parse_yaml_cfg, validate_inputs


# Parse options from yaml file
cfg_file = "./config/config_base.yaml"
cfg = parse_yaml_cfg(cfg_file)

# Inizialize Variables
cams = cfg.paths.cam_names
features = dict.fromkeys(cams)  # @TODO: put this in an inizialization function

# Create Image Datastore objects
images = dict.fromkeys(cams)  # @TODO: put this in an inizialization function
for cam in cams:
    images[cam] = Imageds(cfg.paths.imdir / cam)

cfg = validate_inputs(cfg, images)

""" Perform matching and tracking """
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
        print(f"Processing epoch {epoch}...")

        # opt_matching = cfg.matching.copy()
        epochdir = Path(cfg.paths.resdir) / f"epoch_{epoch}"

        # -- Find Matches at current epoch --#
        print(f"Run Superglue to find matches at epoch {epoch}")
        opt_matching.output_dir = epochdir
        pair = [
            images[cams[0]].get_image_path(epoch),
            images[cams[1]].get_image_path(epoch),
        ]
        # Call matching function
        matchedPts, matchedDescriptors, matchedPtsScores = match_pair(
            pair, cfg.images.bbox, opt_matching
        )

        # Store matches in features structure
        for jj, cam in enumerate(cams):
            # Dict keys are the cameras names, internal list contain epoches
            features[cam].append(Features())
            features[cam][epoch].append_features(
                {
                    "kpts": matchedPts[jj],
                    "descr": matchedDescriptors[jj],
                    "score": matchedPtsScores[jj],
                }
            )
            # @TODO: Store match confidence!

        # === Track previous matches at current epoch ===#
        if cfg.proc.do_tracking and epoch > 0:
            print(f"Track points from epoch {epoch-1} to epoch {epoch}")

            trackoutdir = epochdir / f"from_t{epoch-1}"
            opt_tracking["output_dir"] = trackoutdir
            pairs = [
                [
                    images[cams[0]].get_image_path(epoch - 1),
                    images[cams[0]].get_image_path(epoch),
                ],
                [
                    images[cams[1]].get_image_path(epoch - 1),
                    images[cams[1]].get_image_path(epoch),
                ],
            ]
            prevs = [
                features[cams[0]][epoch - 1].get_features_as_dict(),
                features[cams[1]][epoch - 1].get_features_as_dict(),
            ]
            # Call actual tracking function
            tracked_cam0, tracked_cam1 = track_matches(
                pairs, cfg.images.bbox, prevs, opt_tracking
            )
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
            px_th=1.5,
            conf=0.99999,
            max_iters=10000,
            laf_consistensy_coef=-1.0,
            error_type="sampson",
            symmetric_error_check=True,
            enable_degeneracy_check=True,
        )
        print(
            f"Matching at epoch {epoch}: pydegensac found {inlMask.sum()} \
            inliers ({inlMask.sum()*100/len(features[cams[0]][epoch]):.2f}%)"
        )
        features[cams[0]][epoch].remove_outliers_features(inlMask)
        features[cams[1]][epoch].remove_outliers_features(inlMask)

        # Write matched points to disk
        im_stems = images[cams[0]].get_image_stem(epoch), images[
            cams[1]
        ].get_image_stem(epoch)
        for jj, cam in enumerate(cams):
            features[cam][epoch].save_as_txt(epochdir / f"{im_stems[jj]}_mktps.txt")
        with open(epochdir / f"{im_stems[0]}_{im_stems[1]}_features.pickle", "wb") as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        last_match_path = create_directory("res/last_epoch")
        with open(last_match_path / "last_features.pickle", "wb") as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Matching completed")

elif not features[cams[0]]:
    last_match_path = "res/last_epoch/last_features.pickle"
    with open(last_match_path, "rb") as f:
        features = pickle.load(f)
        print("Loaded previous matches")
else:
    print("Features already present, nothing was changed.")


""" SfM """

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
        images[cams[0]].get_image_stem(epoch) + cfg.georef.target_file_ext
    )

    p2_path = cfg.georef.target_dir / (
        images[cams[1]].get_image_stem(epoch) + cfg.georef.target_file_ext
    )

    targets.append(
        Targets(
            im_file_path=[p1_path, p2_path], obj_file_path="data/target_world_p1.csv"
        )
    )


for epoch in cfg.proc.epoch_to_process:
    # epoch = 0
    print(f"Reconstructing epoch {epoch}...")

    # Initialize Intrinsics
    """ Inizialize Camera Intrinsics at every epoch setting them equal to
        the those of the reference cameras.
    """
    # @TODO: replace append with insert or a more robust data structure...
    for cam in cams:
        cameras[cam].append(
            Camera(
                width=im_width,
                height=im_height,
                calib_path=cfg.paths.caldir / f"{cam}.txt",
            )
        )

    # --- Space resection of Master camera ---#
    # At the first epoch, perform Space resection of the first camera by using GCPs. At all other epoches, set camera 1 EO equal to first one.
    # if epoch == 0:
    #     ''' Initialize Single_camera_geometry class with a cameras object'''
    #     targets_to_use = ['T2','T3','T4','F2' ]
    #     space_resection = Space_resection(cameras[cams[0]][epoch])
    #     space_resection.estimate(
    #         targets[epoch].extract_image_coor_by_label(targets_to_use,cam_id=0),
    #         targets[epoch].extract_object_coor_by_label(targets_to_use)
    #         )
    #     # Store result in camera 0 object
    #     cameras[cams[0]][epoch] = space_resection.camera
    # else:
    #     cameras[cams[0]][epoch] = cameras[cams[0]][0]

    # --- Perform Relative orientation of the two cameras ---#
    """ Initialize Two_view_geometry class with a list containing the two cameras and a list contaning the matched features location on each camera.
    """
    relative_ori = Two_view_geometry(
        [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
        [
            features[cams[0]][epoch].get_keypoints(),
            features[cams[1]][epoch].get_keypoints(),
        ],
    )
    relative_ori.relative_orientation(
        threshold=1.5,
        confidence=0.999999,
        scale_factor=np.linalg.norm(
            cfg.georef.camera_centers_world[0] - cfg.georef.camera_centers_world[1]
        ),
    )
    # Store result in camera 1 object
    cameras[cams[1]][epoch] = relative_ori.cameras[1]

    # --- Triangulate Points ---#
    """ Initialize a Triangulate class instance with a list containing the two cameras and a list contaning the matched features location on each camera.
    Triangulated points are saved as points3d proprierty of the Triangulate object (eg., triangulation.points3d)
    """
    triangulation = Triangulate(
        [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
        [
            features[cams[0]][epoch].get_keypoints(),
            features[cams[1]][epoch].get_keypoints(),
        ],
    )
    points3d = triangulation.triangulate_two_views()
    triangulation.interpolate_colors_from_image(
        images[cams[1]][epoch],
        cameras[cams[1]][epoch],
        convert_BRG2RGB=True,
    )

    # --- Absolute orientation (-> coregistration on stable points) ---#
    targets_to_use = ["F2"]  # 'T4',
    abs_ori = Absolute_orientation(
        (cameras[cams[0]][epoch], cameras[cams[1]][epoch]),
        points3d_final=targets[epoch].extract_object_coor_by_label(targets_to_use),
        image_points=(
            targets[epoch].extract_image_coor_by_label(targets_to_use, cam_id=0),
            targets[epoch].extract_image_coor_by_label(targets_to_use, cam_id=1),
        ),
        camera_centers_world=cfg.georef.camera_centers_world,
    )
    T = abs_ori.estimate_transformation_linear(estimate_scale=True)
    # uncertainty = np.array([
    #     [1., 1., 1.],
    #     [0.001, 0.001, 0.001],
    #     [0.001, 0.001, 0.001],
    #     ])
    # T = abs_ori.estimate_transformation_least_squares(uncertainty=uncertainty)
    points3d = abs_ori.apply_transformation(points3d=points3d)
    for i, cam in enumerate(cams):
        cameras[cam][epoch] = abs_ori.cameras[i]

    # Create point cloud and save .ply to disk
    pcd_epc = create_point_cloud(points3d, triangulation.colors)

    # Filter outliers in point cloud with SOR filter
    # if cfg.other.do_SOR_filter:
    #     _, ind = pcd_epc.remove_statistical_outlier(nb_neighbors=10,
    #                                                 std_ratio=3.0)
    #     #     display_pc_inliers(pcd_epc, ind)
    #     pcd_epc = pcd_epc.select_by_index(ind)
    #     print('Point cloud filtered by Statistical Oulier Removal')

    # Write point cloud to disk and store it in Point Cloud List
    write_ply(pcd_epc, f"res/pt_clouds/sparse_pts_t{epoch}.ply")
    point_clouds.append(pcd_epc)

print("Done.")


# Visualize point cloud
display_point_cloud(
    point_clouds,
    [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
    plot_scale=10,
)


""" Bundle adjustment with Metashape"""
# Export results in Bundler format
write_bundler_out(
    export_dir = "./res/metashape/"
    
)

do_export_to_bundler = True

if do_export_to_bundler:
    #  = Path('res/bundler_output')
 


""" Compute DSM and orthophotos """
# @TODO: implement better DSM class

compute_orthophoto_dsm = False
if compute_orthophoto_dsm:
    print("DSM and orthophoto generation started")
    res = 0.03
    xlim = [-100.0, 80.0]
    ylim = [-10.0, 65.0]

    dsms = []
    ortofoto = dict.fromkeys(cams)
    ortofoto[cams[0]], ortofoto[cams[1]] = [], []
    for epoch in cfg.proc.epoch_to_process:
        print(f"Epoch {epoch}")
        dsms.append(
            build_dsm(
                np.asarray(point_clouds[epoch].points),
                dsm_step=res,
                xlim=xlim,
                ylim=ylim,
                make_dsm_plot=False,
                # fill_value = ,
                # save_path=f'res/dsm/dsm_app_epoch_{epoch}.tif'
            )
        )
        print("DSM built.")
        for cam in cams:
            fout_name = f"res/ortofoto/ortofoto_app_cam_{cam}_epc_{epoch}.tif"
            ortofoto[cam].append(
                generate_ortophoto(
                    cv2.cvtColor(images[cam][epoch], cv2.COLOR_BGR2RGB),
                    dsms[epoch],
                    cameras[cam][epoch],
                    xlim=xlim,
                    ylim=ylim,
                    save_path=fout_name,
                )
            )
        print("Orthophotos built.")
