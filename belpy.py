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
import logging

from pathlib import Path
from copy import deepcopy

# Belpy Classes
from lib.base_classes.camera import Camera
from lib.base_classes.pointCloud import PointCloud
from lib.base_classes.images import Image, Imageds
from lib.base_classes.targets import Targets
from lib.base_classes.features import Features

# Belpy libraries
from lib.matching.matching_base import MatchingAndTracking
from lib.sfm.two_view_geometry import Two_view_geometry
from lib.sfm.triangulation import Triangulate
from lib.sfm.absolute_orientation import (
    Absolute_orientation,
    Space_resection,
)
from lib.read_config import parse_yaml_cfg
from lib.utils.utils import (
    AverageTimer,
    homography_warping,
    build_dsm,
    generate_ortophoto,
)
from lib.utils.inizialize_variables import Inizialization
from lib.visualization import (
    display_point_cloud,
    make_focal_length_variation_plot,
    make_camera_angles_plot,
    plot_features,
    imshow_cv,
)
from lib.import_export.export2bundler import write_bundler_out
from lib.metashape.metashape import (
    MetashapeProject,
    MetashapeReader,
    build_ms_cfg_base,
)


logger = logging.getLogger("Belpy")
timer_global = AverageTimer(newline=True)

# Read options from yaml file
cfg_file = "config/config_base.yaml"
cfg = parse_yaml_cfg(cfg_file, logger)

""" Inizialize Variables """

init = Inizialization(cfg)
init.inizialize_belpy()
cameras = init.cameras
cams = init.cams
features = init.features
images = init.images
targets = init.targets
point_clouds = init.point_clouds
epoch_dict = init.epoch_dict
focals = {0: [], 1: []}

""" Big Loop over epoches """

print("Processing started:")
print("-----------------------")
timer = AverageTimer(newline=True)
for epoch in cfg.proc.epoch_to_process:

    print(f"\nProcessing epoch {epoch}/{cfg.proc.epoch_to_process[-1]}...")

    epochdir = Path(cfg.paths.results_dir) / epoch_dict[epoch]

    # Perform matching and tracking
    if cfg.proc.do_matching:
        features = MatchingAndTracking(
            cfg=cfg,
            epoch=epoch,
            images=images,
            features=features,
            epoch_dict=epoch_dict,
        )
    else:
        try:
            path = epochdir / "matching"
            fname = list(path.glob("*.pickle"))
            if len(fname) < 1:
                raise FileNotFoundError(
                    f"No pickle file found in the epoch directory {epochdir}"
                )
            if len(fname) > 1:
                raise FileNotFoundError(
                    f"More than one pickle file is present in the epoch directory {epochdir}"
                )

            with open(fname[0], "rb") as f:
                try:
                    loaded_features = pickle.load(f)
                except:
                    raise FileNotFoundError(
                        f"Invalid pickle file in epoch directory {epochdir}"
                    )

            for cam, feats in loaded_features.items():
                features[cam].insert(epoch, feats)

            # with open(cfg.paths.last_match_path, "rb") as f:
            #     features = pickle.load(f)
            #     print("Loaded previous matches")

        except FileNotFoundError as err:
            logger.exception(err)

            print("Performing new matching and tracking...")
            features = MatchingAndTracking(
                cfg=cfg,
                epoch=epoch,
                images=images,
                features=features,
                epoch_dict=epoch_dict,
            )

    timer.update("matching")

    """ SfM """

    print(f"Reconstructing epoch {epoch}...")

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
    points3d = triangulation.triangulate_two_views(
        compute_colors=True, image=images[cams[1]][epoch], cam_id=1
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
    pcd_epc = PointCloud(points3d=points3d, points_col=triangulation.colors)
    # point_clouds.insert(epoch, pcd_epc)

    timer.update("relative orientation")

    # Metashape BBA and dense cloud
    if cfg.proc.do_metashape_bba:
        # Export results in Bundler format
        # @TODO: modify function to work for single epoch data
        write_bundler_out(
            export_dir=epochdir / "metashape",
            epoches=[epoch],
            images=images,
            cams=cams,
            cameras=cameras,
            features=features,
            point_cloud=pcd_epc,
            targets=targets,
            targets_to_use=cfg.georef.targets_to_use,
            targets_enabled=[True, True],
        )

        # Temporary function for building configuration dictionary.
        # Must be moved to a file or other solution.
        ms_cfg = build_ms_cfg_base(epochdir, epoch_dict, epoch)
        ms_cfg.build_dense = cfg.proc.do_metashape_dense

        ms = MetashapeProject(ms_cfg, timer)
        ms.process_full_workflow()

        ms_reader = MetashapeReader(
            metashape_dir=epochdir / "metashape",
            num_cams=len(cams),
        )
        ms_reader.read_belpy_outputs()
        for i in range(len(cams)):
            focals[i].insert(epoch, ms_reader.get_focal_lengths()[i])

        # Assign camera extrinsics and intrinsics estimated in Metashape to Camera Object (assignation is done manaully @TODO automatic K and extrinsics matrixes to assign correct camera by camera label)
        new_K = ms_reader.get_K()
        cameras[cams[0]][epoch].update_K(new_K[1])
        cameras[cams[1]][epoch].update_K(new_K[0])

        cameras[cams[0]][epoch].update_extrinsics(
            ms_reader.extrinsics[images[cams[0]].get_image_stem(epoch)]
        )
        cameras[cams[1]][epoch].update_extrinsics(
            ms_reader.extrinsics[images[cams[1]].get_image_stem(epoch)]
        )

        # Triangulate again points and update Point Cloud List
        triangulation = Triangulate(
            [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
            [
                features[cams[0]][epoch].get_keypoints(),
                features[cams[1]][epoch].get_keypoints(),
            ],
        )
        points3d = triangulation.triangulate_two_views(
            compute_colors=True, image=images[cams[1]][epoch], cam_id=1
        )

        # Build new point cloud, save to disk and store it in point_clouds list
        pcd_epc = PointCloud(points3d=points3d, points_col=triangulation.colors)
        pcd_epc.write_ply(
            cfg.paths.results_dir
            / f"point_clouds/sparse_ep_{epoch}_{epoch_dict[epoch]}.ply"
        )
        point_clouds.insert(epoch, pcd_epc)

        # - For debugging purposes
        # M = targets[epoch].extract_object_coor_by_label(cfg.georef.targets_to_use)
        # m = cameras[cams[1]][epoch].project_point(M)
        # plot_features(images[cams[1]][epoch], m)
        # plot_features(images[cams[0]][epoch], features[cams[0]][epoch].get_keypoints())

        # Clean variables
        del relative_ori, triangulation, abs_ori, points3d, pcd_epc
        del T, new_K
        del ms_cfg, ms, ms_reader
        gc.collect()

        cam = "p2"
        image = images[cam][epoch]
        out_path = f"res/warped/{images[cam].get_image_name(epoch)}"
        homography_warping(cameras[cam][0], cameras[cam][epoch], image, out_path, timer)

    timer.print(f"Epoch {epoch} completed")

timer_global.print("Processing completed")


if cfg.other.do_viz:
    # Visualize point cloud
    display_point_cloud(
        point_clouds,
        [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
        plot_scale=10,
    )

    # Display estimated focal length variation
    make_focal_length_variation_plot(focals, "res/focal_lenghts.png")
    make_camera_angles_plot(cameras, "res/angles.png")


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
