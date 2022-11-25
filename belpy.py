"""
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
"""

#%%
import numpy as np
import cv2
import pickle
import gc
from pathlib import Path
from copy import deepcopy

# Classes
from lib.base_classes.camera import Camera
from lib.base_classes.pointCloud import PointCloud
from lib.classes import Imageds, Features, Targets

# Belpy libraries
from lib.matching.matching_base import MatchingAndTracking
from lib.sfm.two_view_geometry import Two_view_geometry
from lib.sfm.triangulation import Triangulate
from lib.sfm.absolute_orientation import (
    Absolute_orientation,
    Space_resection,
)
from lib.read_config import parse_yaml_cfg
from lib.point_clouds import (
    create_point_cloud,
    write_ply,
)
from lib.utils.utils import (
    AverageTimer,
    build_dsm,
    generate_ortophoto,
)
from lib.visualization import (
    display_point_cloud,
    make_focal_length_variation_plot,
    plot_features,
)
from lib.import_export.export2bundler import write_bundler_out
from lib.metashape.metashape import (
    MetashapeProject,
    MetashapeReader,
    build_ms_cfg_base,
)

timer_global = AverageTimer(newline=True)

# Parse options from yaml file
cfg_file = "config/config_base.yaml"
cfg = parse_yaml_cfg(cfg_file)

""" Inizialize Variables """
# @TODO: put this in an inizialization function
cams = cfg.paths.camera_names
features = dict.fromkeys(cams)
cams = cfg.paths.camera_names
for cam in cams:
    features[cam] = []

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
            im_file_path=[p1_path, p2_path],
            obj_file_path=cfg.georef.target_dir / "target_world_p1.csv",
        )
    )

# Cameras
# @TODO: build function for variable inizialization
cameras = dict.fromkeys(cams)
cameras[cams[0]], cameras[cams[1]] = [], []
im_height, im_width = images[cams[0]][0].shape[:2]
# @TODO: store this information in exif inside an Image Class
point_clouds = []

# Tmp variable to store only estimated focal lenghts in Metashape
focals = {0: [], 1: []}

from lib.base_classes.images import Image

epoch_dir_dict = {}
for epoch in cfg.proc.epoch_to_process:
    image = Image(images[cams[0]].get_image_path(epoch))
    epoch_dir_dict[epoch] = Path(
        (cfg.paths.results_dir)
        / f"{image.date.year}_{image.date.month:02}_{image.date.day:02}"
    ).stem


""" Big Loop over epoches """

print("Processing started:")
print("-----------------------")
timer = AverageTimer(newline=True)
for epoch in cfg.proc.epoch_to_process:

    print(f"\nProcessing epoch {epoch}...")

    epochdir = Path(cfg.paths.results_dir) / epoch_dir_dict[epoch]
    # epochdir = Path(cfg.paths.results_dir) / f"epoch_{epoch}"

    # Perform matching and tracking
    if cfg.proc.do_matching:
        features = MatchingAndTracking(
            cfg=cfg,
            epoch=epoch,
            images=images,
            features=features,
            epoch_dict=epoch_dir_dict,
            # res_dir=epochdir,
            # prev_epoch_dir=epoch_dir_dict[epoch-1],
        )
    elif not features[cams[0]]:
        try:
            with open(cfg.paths.last_match_path, "rb") as f:
                features = pickle.load(f)
                print("Loaded previous matches")
        except:
            print(
                f"Features not found in {str(cfg.paths.last_match_path)}. Please enable performing matching or provide valid path to already computed matches."
            )
    else:
        print("Features already loaded.")
    timer.update("matching")

    """ SfM """

    print(f"Reconstructing epoch {epoch}...")

    # Initialize Intrinsics
    # Inizialize Camera Intrinsics at every epoch setting them equal to the those of the reference cameras.
    # @TODO: replace append with insert or a more robust data structure...
    for cam in cams:
        cameras[cam].insert(
            epoch,
            Camera(
                width=im_width,
                height=im_height,
                calib_path=cfg.paths.calibration_dir / f"{cam}.txt",
            ),
        )

    # --- Space resection of Master camera ---#
    # At the first epoch, perform Space resection of the first camera by using GCPs. At all other epoches, set camera 1 EO equal to first one.
    if cfg.proc.do_space_resection and epoch == 0:
        """Initialize Single_camera_geometry class with a cameras object"""
        cfg.georef.targets_to_use = ["T2", "T3", "T4", "F2", "F4"]
        space_resection = Space_resection(cameras[cams[0]][epoch])
        space_resection.estimate(
            targets[epoch].extract_image_coor_by_label(
                cfg.georef.targets_to_use, cam_id=0
            ),
            targets[epoch].extract_object_coor_by_label(cfg.georef.targets_to_use),
        )
        # Store result in camera 0 object
        cameras[cams[0]][epoch] = space_resection.camera

    # --- Perform Relative orientation of the two cameras ---#
    # Initialize Two_view_geometry class with a list containing the two cameras and a list contaning the matched features location on each camera.
    relative_ori = Two_view_geometry(
        [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
        [
            features[cams[0]][epoch].get_keypoints(),
            features[cams[1]][epoch].get_keypoints(),
        ],
    )
    relative_ori.relative_orientation(
        threshold=cfg.other.pydegensac_treshold,
        confidence=0.999999,
        scale_factor=np.linalg.norm(
            cfg.georef.camera_centers_world[0] - cfg.georef.camera_centers_world[1]
        ),
    )
    # Store result in camera 1 object
    cameras[cams[1]][epoch] = relative_ori.cameras[1]

    # --- Triangulate Points ---#
    # Initialize a Triangulate class instance with a list containing the two cameras and a list contaning the matched features location on each camera. Triangulated points are saved as points3d proprierty of the Triangulate object (eg., triangulation.points3d)
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
    if cfg.proc.do_coregistration:
        abs_ori = Absolute_orientation(
            (cameras[cams[0]][epoch], cameras[cams[1]][epoch]),
            points3d_final=targets[epoch].extract_object_coor_by_label(
                cfg.georef.targets_to_use
            ),
            image_points=(
                targets[epoch].extract_image_coor_by_label(
                    cfg.georef.targets_to_use, cam_id=0
                ),
                targets[epoch].extract_image_coor_by_label(
                    cfg.georef.targets_to_use, cam_id=1
                ),
            ),
            camera_centers_world=cfg.georef.camera_centers_world,
        )
        T = abs_ori.estimate_transformation_linear(estimate_scale=True)
        points3d = abs_ori.apply_transformation(points3d=points3d)
        for i, cam in enumerate(cams):
            cameras[cam][epoch] = abs_ori.cameras[i]

    # Create point cloud and save .ply to disk
    pcd_epc = PointCloud(points3d, triangulation.colors)
    # pcd_epc.write_ply(epochdir / f"sparse_pts_t{epoch}.ply")

    point_clouds.insert(epoch, pcd_epc)

    timer.update("relative orientation")

    # Metashape BBA and dense cloud
    if cfg.proc.do_metashape_bba:
        # Export results in Bundler format
        write_bundler_out(
            export_dir=epochdir / "metashape",
            epoches=[epoch],
            images=deepcopy(images),
            cams=cams,
            cameras=deepcopy(cameras),
            features=deepcopy(features),
            point_clouds=deepcopy(point_clouds),
            targets=deepcopy(targets),
            targets_to_use=cfg.georef.targets_to_use,
            targets_enabled=[True, True],
        )

        # Temporary function for building configuration dictionary.
        # Must be moved to a file or other solution.
        ms_cfg = build_ms_cfg_base(epochdir, epoch)
        ms_cfg.build_dense = cfg.proc.do_metashape_dense

        ms = MetashapeProject(ms_cfg, timer)
        ms.process_full_workflow()

        ms_reader = MetashapeReader(
            metashape_dir=epochdir / "metashape",
            num_cams=len(cams),
        )
        ms_reader.read_calibration_from_file()
        ms_reader.read_cameras_from_file(
            epochdir / f"metashape/belpy_epoch_{epoch}_camera_estimated.txt"
        )
        for i in range(len(cams)):
            focals[i].insert(epoch, ms_reader.get_focal_lengths()[i])

        # Assign camera extrinsics and intrinsics estimated in Metashape to Camera Object (assignation is done manaully @TODO automatic K and extrinsics matrixes to assign correct camera by camera label)

        # new_K = ms_reader.get_K()
        # camera0 = deepcopy(cameras[cams[0]][epoch])
        # camera1 = deepcopy(cameras[cams[1]][epoch])

        # camera0.update_K(new_K[1])
        # camera1.update_K(new_K[0])
        # new_extrinsics = ms_reader.get_extrinsics()
        # camera0.update_extrinsics(new_extrinsics[1])
        # camera1.update_extrinsics(new_extrinsics[0])

        # # Triangulate again points and update Point Cloud List
        # triangulation = Triangulate(
        #     [camera0, camera1],
        #     [
        #         features[cams[0]][epoch].get_keypoints(),
        #         features[cams[1]][epoch].get_keypoints(),
        #     ],
        # )
        # points3d = triangulation.triangulate_two_views()
        # triangulation.interpolate_colors_from_image(
        #     images[cams[1]][epoch],
        #     camera1,
        # )
        # new_pcd = PointCloud(points3d, triangulation.colors)
        # new_pcd.write_ply(f"res/point_clouds/sparse_pts_t{epoch}.ply")

        # M = targets[epoch].extract_object_coor_by_label(cfg.georef.targets_to_use)
        # m = camera1.project_point(M)
        # plot_features(images[cams[1]][epoch], m)

        # plot_features(images[cams[0]][epoch], features[cams[0]][epoch].get_keypoints())

        # display_point_cloud(
        #     new_pcd,  # new_pcd, #point_clouds
        #     [camera0, camera1],
        #     plot_scale=20,
        # )

        # Clean variables
        # del relative_ori, triangulation, abs_ori, points3d, pcd_epc
        # del T, new_K, new_extrinsics
        # del ms_cfg, ms, ms_reader
        # gc.collect()

    timer.print(f"Epoch {epoch} completed")


timer_global.print("Processing completed")


# import open3d as o3d

# epoch = 0
# # pcd = o3d.io.read_point_cloud("res/point_clouds/dense_epoch_0.ply")
# pcd_uav = o3d.io.read_point_cloud("tmp.ply")
# display_point_cloud(
#     [pcd_uav],
#     [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
#     viz_rs=False,
#     plot_scale=20,
# )

if cfg.other.do_viz:
    # Visualize point cloud
    display_point_cloud(
        point_clouds,
        [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
        plot_scale=10,
    )

    # Display estimated focal length variation
    make_focal_length_variation_plot(focals, "res/focal_lenghts.png")


#%%
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
