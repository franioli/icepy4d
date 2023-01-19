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
import shutil

from pathlib import Path

# Belpy Classes
from lib.base_classes.camera import Camera
from lib.base_classes.pointCloud import PointCloud
from lib.base_classes.images import Image, ImageDS
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
from lib.utils.initialization import parse_yaml_cfg, Inizialization
from lib.utils.utils import (
    AverageTimer,
    homography_warping,
    build_dsm,
    generate_ortophoto,
)
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
    build_metashape_cfg,
)

print("\n===========================================================")
print("Belpy")
print("Low-cost stereo photogrammetry for 4D glacier monitoring ")
print("2022 - Francesco Ioli - francesco.ioli@polimi.it")
print("===========================================================\n")

CFG_FILE = "config/config_block_3_4.yaml"

# Create logger and set logging level
LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | '%(filename)s -> %(funcName)s', line %(lineno)d - %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)
logger = logging.getLogger(__name__)

# Read options from yaml file
cfg_file = Path(CFG_FILE)
logger.info(f"Configuration file: {cfg_file.stem}")
timer_global = AverageTimer(newline=True)
cfg = parse_yaml_cfg(cfg_file)

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
focals = init.focals_dict

""" Big Loop over epoches """
print("\nProcessing started:")
print("-----------------------")
timer = AverageTimer(newline=True)
iter = 0
for epoch in cfg.proc.epoch_to_process:

    logger.info(
        f"\nProcessing epoch {epoch} [{iter}/{cfg.proc.epoch_to_process[-1]-cfg.proc.epoch_to_process[0]}] - {epoch_dict[epoch]}..."
    )
    iter += 1

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
                    features[epoch] = loaded_features
                except:
                    raise FileNotFoundError(
                        f"Invalid pickle file in epoch directory {epochdir}"
                    )

        except FileNotFoundError as err:
            logger.exception(err)
            logger.info("Performing new matching and tracking...")
            features = MatchingAndTracking(
                cfg=cfg,
                epoch=epoch,
                images=images,
                features=features,
                epoch_dict=epoch_dict,
            )

    timer.update("matching")

    """ SfM """

    logger.info(f"Reconstructing epoch {epoch}...")

    # --- Space resection of Master camera ---#
    # At the first epoch, perform Space resection of the first camera by using GCPs. At all other epoches, set camera 1 EO equal to first one.
    if cfg.proc.do_space_resection and epoch == 0:
        """Initialize Single_camera_geometry class with a cameras object"""
        space_resection = Space_resection(cameras[epoch][cams[0]])
        space_resection.estimate(
            targets[epoch].get_image_coor_by_label(cfg.georef.targets_to_use, cam_id=0),
            targets[epoch].get_object_coor_by_label(cfg.georef.targets_to_use),
        )
        # Store result in camera 0 object
        cameras[epoch][cams[0]] = space_resection.camera

    # --- Perform Relative orientation of the two cameras ---#
    # Initialize Two_view_geometry class with a list containing the two cameras and a list contaning the matched features location on each camera.
    relative_ori = Two_view_geometry(
        [cameras[epoch][cams[0]], cameras[epoch][cams[1]]],
        [
            features[epoch][cams[0]].get_keypoints(),
            features[epoch][cams[1]].get_keypoints(),
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
    cameras[epoch][cams[1]] = relative_ori.cameras[1]
    logger.info("Relative orientation completed.")

    # --- Triangulate Points ---#
    # Initialize a Triangulate class instance with a list containing the two cameras and a list contaning the matched features location on each camera. Triangulated points are saved as points3d proprierty of the Triangulate object (eg., triangulation.points3d)
    triangulation = Triangulate(
        [cameras[epoch][cams[0]], cameras[epoch][cams[1]]],
        [
            features[epoch][cams[0]].get_keypoints(),
            features[epoch][cams[1]].get_keypoints(),
        ],
    )
    points3d = triangulation.triangulate_two_views(
        compute_colors=True, image=images[cams[1]].read_image(epoch).value, cam_id=1
    )
    logger.info("Tie points triangulated.")

    # --- Absolute orientation (-> coregistration on stable points) ---#
    if cfg.proc.do_coregistration:
        try:
            abs_ori = Absolute_orientation(
                (cameras[epoch][cams[0]], cameras[epoch][cams[1]]),
                points3d_final=targets[epoch].get_object_coor_by_label(
                    cfg.georef.targets_to_use
                ),
                image_points=(
                    targets[epoch].get_image_coor_by_label(
                        cfg.georef.targets_to_use, cam_id=0
                    ),
                    targets[epoch].get_image_coor_by_label(
                        cfg.georef.targets_to_use, cam_id=1
                    ),
                ),
                camera_centers_world=cfg.georef.camera_centers_world,
            )
            T = abs_ori.estimate_transformation_linear(estimate_scale=True)
            points3d = abs_ori.apply_transformation(points3d=points3d)
            for i, cam in enumerate(cams):
                cameras[epoch][cam] = abs_ori.cameras[i]
            logger.info("Absolute orientation completed.")
        except ValueError as err:
            logger.error(
                "Absolute orientation not succeded. Not enough targets available. Skipping to the next epoch."
            )
            continue

    # Create point cloud and save .ply to disk
    pcd_epc = PointCloud(points3d=points3d, points_col=triangulation.colors)

    timer.update("relative orientation")

    # Metashape BBA and dense cloud
    if cfg.proc.do_metashape_processing:
        # Export results in Bundler format

        # If a metashape folder is already present, delete it completely and start a new metashape project
        metashape_path = epochdir / "metashape"
        if metashape_path.exists() and cfg.metashape.force_overwrite_projects:
            logging.warning(
                f"Metashape folder {metashape_path} already exists, but force_overwrite_projects is set to True. Removing all old Metashape files"
            )
            shutil.rmtree(metashape_path, ignore_errors=True)

        im_dict = {cam: images[cam].get_image_path(epoch) for cam in cams}
        write_bundler_out(
            export_dir=epochdir,
            im_dict=im_dict,
            cams=cams,
            cameras=cameras[epoch],
            features=features[epoch],
            point_cloud=pcd_epc,
            targets=targets[epoch],
            targets_to_use=cfg.georef.targets_to_use,
            targets_enabled=[True for el in cfg.georef.targets_to_use],
        )

        ms_cfg = build_metashape_cfg(cfg, epoch_dict, epoch)
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
        cameras[epoch][cams[0]].update_K(new_K[1])
        cameras[epoch][cams[1]].update_K(new_K[0])

        cameras[epoch][cams[0]].update_extrinsics(
            ms_reader.extrinsics[images[cams[0]].get_image_stem(epoch)]
        )
        cameras[epoch][cams[1]].update_extrinsics(
            ms_reader.extrinsics[images[cams[1]].get_image_stem(epoch)]
        )

        # Triangulate again points and update Point Cloud dict
        triangulation = Triangulate(
            [cameras[epoch][cams[0]], cameras[epoch][cams[1]]],
            [
                features[epoch][cams[0]].get_keypoints(),
                features[epoch][cams[1]].get_keypoints(),
            ],
        )
        points3d = triangulation.triangulate_two_views(
            compute_colors=True, image=images[cams[1]].read_image(epoch).value, cam_id=1
        )

        pcd_epc = PointCloud(points3d=points3d, points_col=triangulation.colors)
        if cfg.proc.save_sparse_cloud:
            pcd_epc.write_ply(
                cfg.paths.results_dir / f"point_clouds/sparse_{epoch_dict[epoch]}.ply"
            )
        point_clouds[epoch] = pcd_epc

        # - For debugging purposes
        # M = targets[epoch].get_object_coor_by_label(cfg.georef.targets_to_use)
        # m = cameras[epoch][cams[1]].project_point(M)
        # plot_features(images[cams[1]].read_image(epoch).value, m)
        # plot_features(images[cams[0]].read_image(epoch).value, features[epoch][cams[0]].get_keypoints())

        # Clean variables
        del relative_ori, triangulation, abs_ori, points3d, pcd_epc
        del T, new_K
        del ms_cfg, ms, ms_reader
        gc.collect()

        # Homograpghy warping
        if cfg.proc.do_homography_warping:
            ep_ini = cfg.proc.epoch_to_process[0]
            cam = cfg.proc.camera_to_warp
            image = images[cams[1]].read_image(epoch).value
            out_path = f"res/warped/{images[cam][epoch]}"
            homography_warping(
                cameras[ep_ini][cam], cameras[epoch][cam], image, out_path, timer
            )

        # Incremetal plots
        # if cfg.other.do_viz:
        #     make_focal_length_variation_plot(
        #         focals,
        #         cfg.paths.results_dir / f"focal_lenghts_{cfg_file.stem}.png",
        #     )
        #     make_camera_angles_plot(
        #         cameras,
        #         cfg.paths.results_dir / f"angles_{cfg_file.stem}.png",
        #         baseline_epoch=cfg.proc.epoch_to_process[0],
        #         current_epoch=epoch,
        #     )

    timer.print(f"Epoch {epoch} completed")


timer_global.update("SfM")


if cfg.other.do_viz:
    # Visualize point cloud
    display_point_cloud(
        point_clouds,
        [cameras[epoch][cams[0]], cameras[epoch][cams[1]]],
        plot_scale=10,
    )

    # Display estimated focal length variation
    make_focal_length_variation_plot(
        focals, cfg.paths.results_dir / f"focal_lenghts_{cfg_file.stem}.png"
    )
    make_camera_angles_plot(
        cameras,
        cfg.paths.results_dir / f"angles_{cfg_file.stem}.png",
        baseline_epoch=cfg.proc.epoch_to_process[0],
    )


timer_global.update("Visualization")
timer_global.print("Processing completed")

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
                    cv2.cvtColor(
                        images[cam].read_image(epoch).value, cv2.COLOR_BGR2RGB
                    ),
                    dsms[epoch],
                    cameras[epoch][cam],
                    xlim=xlim,
                    ylim=ylim,
                    save_path=fout_name,
                )
            )
        print("Orthophotos built.")
