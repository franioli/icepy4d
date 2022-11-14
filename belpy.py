import numpy as np
import cv2
import pickle
from copy import deepcopy
from shutil import copy as scopy

from lib.classes import Camera, Imageds, Features, Targets
from lib.matching.matching_base import MatchingAndTracking
from lib.sfm.two_view_geometry import Two_view_geometry
from lib.sfm.triangulation import Triangulate
from lib.sfm.absolute_orientation import (
    Absolute_orientation,
    Space_resection,
)
from lib.config import parse_yaml_cfg
from lib.point_clouds import (
    create_point_cloud,
    write_ply,
)
from lib.utils import (
    create_directory,
    build_dsm,
    generate_ortophoto,
)
from lib.visualization import display_point_cloud

from thirdparty.transformations import euler_from_matrix, euler_matrix


# Parse options from yaml file
cfg_file = "./config/config_base.yaml"
cfg = parse_yaml_cfg(cfg_file)


""" Inizialize Variables """
# @TODO: put this in an inizialization function
cams = cfg.paths.camera_names
features = dict.fromkeys(cams)

# Create Image Datastore objects
images = dict.fromkeys(cams)
for cam in cams:
    images[cam] = Imageds(cfg.paths.image_dir / cam)

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

# Cameras
# @TODO: build function for variable inizialization
cameras = dict.fromkeys(cams)
cameras[cams[0]], cameras[cams[1]] = [], []
im_height, im_width = 4000, 6000
# @TODO: store this information in exif inside an Image Class
point_clouds = []


""" Perform matching and tracking """
if cfg.proc.do_matching:
    MatchingAndTracking(
        cfg=cfg,
        images=images,
        features=features,
    )
# features =
elif not features[cams[0]]:
    last_match_path = "res/last_epoch/last_features.pickle"
    with open(last_match_path, "rb") as f:
        features = pickle.load(f)
        print("Loaded previous matches")
else:
    print("Features already present.")


""" SfM """

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
                calib_path=cfg.paths.calibration_dir / f"{cam}.txt",
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
# write_bundler_out(
#     export_dir = "./res/metashape/"

# )

# do_export_to_bundler = True

# if do_export_to_bundler:
#     #  = Path('res/bundler_output')


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
