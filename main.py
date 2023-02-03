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
import gc
import logging
import shutil

from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime

# icepy classes
import src.icepy.classes as icepy_classes

# icepy libraries
import src.icepy.sfm as sfm
import src.icepy.metashape.metashape as MS
import src.icepy.utils.initialization as initialization
import src.icepy.utils as icepy_utils
import src.icepy.visualization as icepy_viz

from src.icepy.matching.matching_base import (
    MatchingAndTracking,
    load_matches_from_disk,
)
from src.icepy.utils.utils import homography_warping
from src.icepy.io.export2bundler import write_bundler_out


if __name__ == "__main__":

    # Define some global parameters
    # CFG_FILE = "config/config_block_3_4.yaml"
    # cfg_file = Path(CFG_FILE)

    cfg_file, log_cfg = initialization.parse_command_line()

    # cfg_file = Path("config/config_test.yaml")

    # Setup logger
    icepy_utils.setup_logger(
        log_cfg["log_folder"],
        log_cfg["log_name"],
        log_cfg["log_file_level"],
        log_cfg["log_console_level"],
    )

    print("\n===========================================================")
    print("ICEpy4D")
    print(
        "Image-based Continuos monitoring of glaciers' Evolution with low-cost stereo-cameras and Deep Learning photogrammetry"
    )
    print("2022 - Francesco Ioli - francesco.ioli@polimi.it")
    print("===========================================================\n")

    # Read options from yaml file
    logging.info(f"Configuration file: {cfg_file.stem}")
    timer_global = icepy_utils.AverageTimer()
    cfg = initialization.parse_yaml_cfg(cfg_file)

    """ Inizialize Variables """

    init = initialization.Inizialization(cfg)
    init.inizialize_icepy()
    cameras = init.cameras
    cams = init.cams
    features = init.features
    images = init.images
    targets = init.targets
    point_clouds = init.point_clouds
    epoch_dict = init.epoch_dict
    focals = init.focals_dict

    """ Big Loop over epoches """

    logging.info("------------------------------------------------------")
    logging.info("Processing started:")
    timer = icepy_utils.AverageTimer()
    iter = 0
    for epoch in cfg.proc.epoch_to_process:

        logging.info("------------------------------------------------------")
        logging.info(
            f"Processing epoch {epoch} [{iter}/{cfg.proc.epoch_to_process[-1]-cfg.proc.epoch_to_process[0]}] - {epoch_dict[epoch]}..."
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
                features[epoch] = load_matches_from_disk()
            except FileNotFoundError as err:
                logging.exception(err)
                logging.warning("Performing new matching and tracking...")
                features = MatchingAndTracking(
                    cfg=cfg,
                    epoch=epoch,
                    images=images,
                    features=features,
                    epoch_dict=epoch_dict,
                )

        timer.update("matching")

        # Testing
        # a = features[180]["p1"]
        # b = features[181]["p1"]

        """ SfM """

        logging.info(f"Reconstructing epoch {epoch}...")

        # --- Space resection of Master camera ---#
        # At the first epoch, perform Space resection of the first camera by using GCPs. At all other epoches, set camera 1 EO equal to first one.
        if cfg.proc.do_space_resection and epoch == 0:
            """Initialize Single_camera_geometry class with a cameras object"""
            space_resection = abs_ori.Space_resection(cameras[epoch][cams[0]])
            space_resection.estimate(
                targets[epoch].get_image_coor_by_label(
                    cfg.georef.targets_to_use, cam_id=0
                )[0],
                targets[epoch].get_object_coor_by_label(cfg.georef.targets_to_use)[0],
            )
            # Store result in camera 0 object
            cameras[epoch][cams[0]] = space_resection.camera

        # --- Perform Relative orientation of the two cameras ---#
        # Initialize RelativeOrientation class with a list containing the two cameras and a list contaning the matched features location on each camera.
        relative_ori = sfm.RelativeOrientation(
            [cameras[epoch][cams[0]], cameras[epoch][cams[1]]],
            [
                features[epoch][cams[0]].kpts_to_numpy(),
                features[epoch][cams[1]].kpts_to_numpy(),
            ],
        )
        relative_ori.estimate_pose(
            threshold=cfg.other.pydegensac_treshold,
            confidence=0.999999,
            scale_factor=np.linalg.norm(
                cfg.georef.camera_centers_world[0] - cfg.georef.camera_centers_world[1]
            ),
        )
        # Store result in camera 1 object
        cameras[epoch][cams[1]] = relative_ori.cameras[1]

        # --- Triangulate Points ---#
        # Initialize a Triangulate class instance with a list containing the two cameras and a list contaning the matched features location on each camera. Triangulated points are saved as points3d proprierty of the Triangulate object (eg., triangulation.points3d)
        triang = sfm.Triangulate(
            [cameras[epoch][cams[0]], cameras[epoch][cams[1]]],
            [
                features[epoch][cams[0]].kpts_to_numpy(),
                features[epoch][cams[1]].kpts_to_numpy(),
            ],
        )
        points3d = triang.triangulate_two_views(
            compute_colors=True, image=images[cams[1]].read_image(epoch).value, cam_id=1
        )
        logging.info("Tie points triangulated.")

        # --- Absolute orientation (-> coregistration on stable points) ---#
        if cfg.proc.do_coregistration:

            # Get targets available in all cameras
            # Labels of valid targets are returned as second element by get_image_coor_by_label() method
            valid_targets = targets[epoch].get_image_coor_by_label(
                cfg.georef.targets_to_use, cam_id=0
            )[1]
            for id in range(1, len(cams)):
                assert (
                    valid_targets
                    == targets[epoch].get_image_coor_by_label(
                        cfg.georef.targets_to_use, cam_id=id
                    )[1]
                ), f"Epoch {epoch} - {epoch_dict[epoch]}: Different targets found in image {id} - {images[cams[id]][epoch]}"
            if len(valid_targets) < 1:
                logging.error(
                    f"Not enough targets found. Skipping epoch {epoch} and moving to next epoch"
                )
                continue
            if valid_targets != cfg.georef.targets_to_use:
                logging.warning(f"Not all targets found. Using onlys {valid_targets}")

            image_coords = [
                targets[epoch].get_image_coor_by_label(valid_targets, cam_id=id)[0]
                for id, cam in enumerate(cams)
            ]
            obj_coords = targets[epoch].get_object_coor_by_label(valid_targets)[0]
            try:
                abs_ori = sfm.Absolute_orientation(
                    (cameras[epoch][cams[0]], cameras[epoch][cams[1]]),
                    points3d_final=obj_coords,
                    image_points=image_coords,
                    camera_centers_world=cfg.georef.camera_centers_world,
                )
                T = abs_ori.estimate_transformation_linear(estimate_scale=True)
                points3d = abs_ori.apply_transformation(points3d=points3d)
                for i, cam in enumerate(cams):
                    cameras[epoch][cam] = abs_ori.cameras[i]
                logging.info("Absolute orientation completed.")
            except ValueError as err:
                logging.error(err)
                logging.error(
                    f"Absolute orientation not succeded. Not enough targets available. Skipping epoch {epoch} and moving to next epoch"
                )
                continue

        # Create point cloud and save .ply to disk
        pcd_epc = icepy_classes.PointCloud(points3d=points3d, points_col=triang.colors)

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
                targets_to_use=valid_targets,
                targets_enabled=[True for el in valid_targets],
            )

            ms_cfg = MS.build_metashape_cfg(cfg, epoch_dict, epoch)
            ms = MS.MetashapeProject(ms_cfg, timer)
            ms.process_full_workflow()

            ms_reader = MS.MetashapeReader(
                metashape_dir=epochdir / "metashape",
                num_cams=len(cams),
            )
            ms_reader.read_icepy_outputs()
            for i, cam in enumerate(cams):
                focals[cam][epoch] = ms_reader.get_focal_lengths()[i]

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
            triang = sfm.Triangulate(
                [cameras[epoch][cams[0]], cameras[epoch][cams[1]]],
                [
                    features[epoch][cams[0]].kpts_to_numpy(),
                    features[epoch][cams[1]].kpts_to_numpy(),
                ],
            )
            points3d = triang.triangulate_two_views(
                compute_colors=True,
                image=images[cams[1]].read_image(epoch).value,
                cam_id=1,
            )

            pcd_epc = icepy_classes.PointCloud(
                points3d=points3d, points_col=triang.colors
            )
            if cfg.proc.save_sparse_cloud:
                pcd_epc.write_ply(
                    cfg.paths.results_dir
                    / f"point_clouds/sparse_{epoch_dict[epoch]}.ply"
                )
            point_clouds[epoch] = pcd_epc

            # - For debugging purposes
            # M = targets[epoch].get_object_coor_by_label(cfg.georef.targets_to_use)[0]
            # m = cameras[epoch][cams[1]].project_point(M)
            # plot_features(images[cams[1]].read_image(epoch).value, m)
            # plot_features(images[cams[0]].read_image(epoch).value, features[epoch][cams[0]].kpts_to_numpy())

            # Clean variables
            del relative_ori, triang, abs_ori, points3d, pcd_epc
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

        timer.print(f"Epoch {epoch} completed")

    timer_global.update("SfM")

    # Check estimated focal lenghts:
    if cfg.proc.do_metashape_processing:
        logging.info("Checking estimated Focal Lenghts...")
        max_f_variation = 10  # [px]
        quantile_limits = [0.01, 0.99]
        for cam in cams:
            f_median = np.median(list(focals[cam].values()))
            qf = np.quantile(list(focals[cam].values()), quantile_limits)
            for k, v in focals[cam].items():
                if abs(v - f_median) > max_f_variation:
                    logging.warning(
                        f"Focal lenght estimated at epoch {k} ({epoch_dict[k]}) for camera {cam} is has a difference from the median focal lenght larger than {max_f_variation} (estimated: {v:.2f} - median: {f_median:.2f}). Check carefully the results of epoch {epoch_dict[k]}!"
                    )
                if v < qf[0] or v > qf[1]:
                    logging.warning(
                        f"Focal lenght estimated at epoch {k} ({epoch_dict[k]}) for camera {cam} is outside the range between quantile {quantile_limits[0]} and {quantile_limits[1]} of the distribution (estimated: {v:.2f} limits: {qf[0]:.2f} - {qf[1]:.2f}). Check carefully the results of epoch {epoch_dict[k]}!"
                    )

    if cfg.other.do_viz:
        """Put this code into functions in visualization module of icepy"""

        # Visualize point cloud
        # display_point_cloud(
        #     point_clouds,
        #     [cameras[epoch][cams[0]], cameras[epoch][cams[1]]],
        #     plot_scale=10,
        # )

        fig, ax = plt.subplots(1, len(cams))
        for s_id, cam in enumerate(cams):
            ax[s_id].hist(list(focals[cam].values()), density=True)
            ax[s_id].grid(visible=True)
            ax[s_id].set_ylabel("density")
            ax[s_id].set_xlabel("Focal lenght [px]")
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(
            cfg.paths.results_dir / f"focal_lenghts_hist_{cfg_file.stem}.png",
            dpi=100,
        )

        dates = [epoch_dict[ep] for ep in cfg.proc.epoch_to_process]
        dates = [datetime.strptime(date, "%Y_%m_%d") for date in dates]
        fig, ax = plt.subplots(1, len(cams))
        fig.autofmt_xdate()
        for s_id, cam in enumerate(cams):
            ax[s_id].plot(dates, list(focals[cam].values()), "o")
            ax[s_id].grid(visible=True, which="both")
            ax[s_id].set_xlabel("Epoch")
            ax[s_id].set_ylabel("Focal lenght [px]")
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(
            cfg.paths.results_dir / f"focal_lenghts_{cfg_file.stem}.png", dpi=100
        )

        # Make function again
        # make_camera_angles_plot(
        #     cameras,
        #     cfg.paths.results_dir / f"angles_{cfg_file.stem}.png",
        #     baseline_epoch=cfg.proc.epoch_to_process[0],
        # )

    timer_global.update("Visualization")
    timer_global.print("Processing completed")

    #%%
    """ Compute DSM and orthophotos """
    # @TODO: implement better DSM class

    compute_orthophoto_dsm = False
    if compute_orthophoto_dsm:

        from icepy.utils.dsm_orthophoto import build_dsm, generate_ortophoto

        logging.info("DSM and orthophoto generation started")
        res = 0.03
        xlim = [-100.0, 80.0]
        ylim = [-10.0, 65.0]

        dsms = []
        ortofoto = dict.fromkeys(cams)
        ortofoto[cams[0]], ortofoto[cams[1]] = [], []
        for epoch in cfg.proc.epoch_to_process:
            logging.info(f"Epoch {epoch}")
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
            logging.info("DSM built.")
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
            logging.info("Orthophotos built.")
