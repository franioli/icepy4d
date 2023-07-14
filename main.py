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

import gc
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# icepy4d4D
import icepy4d.classes as icepy4d_classes
import icepy4d.metashape.metashape as MS
import icepy4d.sfm as sfm
import icepy4d.utils as icepy4d_utils
import icepy4d.utils.initialization as inizialization
import icepy4d.visualization as icepy4d_viz
from icepy4d.classes.epoch import Epoch, Epoches
from icepy4d.io.export2bundler import write_bundler_out
from icepy4d.matching.match_by_preselection import (
    find_matches_on_patches,
    match_by_preselection,
)
from icepy4d.matching.matching_base import MatchingAndTracking
from icepy4d.matching.tracking_base import tracking_base
from icepy4d.matching.utils import geometric_verification, load_matches_from_disk

# Temporary parameters TODO: put them in config file
CFG_FILE = "config/config_2022.yaml"
LOAD_EXISTING_SOLUTION = False
DO_PRESELECTION = False
DO_ADDITIONAL_MATCHING = True
PATCHES = [
    {"p1": [0, 500, 2000, 2000], "p2": [4000, 0, 6000, 1500]},
    {"p1": [1000, 1500, 4500, 2500], "p2": [1500, 1500, 5000, 2500]},
    {"p1": [2000, 2000, 3000, 3000], "p2": [2100, 2100, 3100, 3100]},
    {"p1": [2300, 1700, 3300, 2700], "p2": [3000, 1900, 4000, 2900]},
]
# TODO: parse_yaml_cfg set deafults paths to results file, check this.


def write_cameras_to_disk(fname, epoch, date, sep=","):
    from icepy4d.thirdparty.transformations import euler_from_matrix

    if epoch is None:
        return

    if not Path(fname).exists():
        items = [
            "date",
            "f1",
            "omega1",
            "phi1",
            "kappa1",
            "f2",
            "omega2",
            "phi2",
            "kappa2",
        ]
        with open(cfg.camea_estimated_fname, "w") as file:
            file.write(f"{f'{sep}'.join(items)}\n")

    with open(fname, "a") as file:
        file.write(f"{date}")
        for cam in epoch.cameras.keys():
            f = epoch.cameras[cam].K[1, 1]
            # R = epoch.cameras[cam].R
            R = epoch.cameras[cam].pose[:3, :3]
            o, p, k = euler_from_matrix(R)
            o, p, k = np.rad2deg(o), np.rad2deg(p), np.rad2deg(k)
            file.write(f"{sep}{f:.2f}{sep}{o:.4f}{sep}{p:.4f}{sep}{k:.4f}")
        file.write("\n")


def compute_reprojection_error(fname, epoch, sep=","):
    print("Computing reprojection error")

    import pandas as pd

    residuals = pd.DataFrame()
    for cam_key, camera in epoch.cameras.items():
        feat = epoch.features[cam_key]
        projections = camera.project_point(epoch.points.to_numpy())
        res = projections - feat.kpts_to_numpy()
        res_norm = np.linalg.norm(res, axis=1)
        residuals["track_id"] = feat.get_track_ids()
        residuals[f"x_{cam_key}"] = res[:, 0]
        residuals[f"y_{cam_key}"] = res[:, 1]
        residuals[f"norm_{cam_key}"] = res_norm

    # Compute global norm as mean of all cameras
    residuals[f"global_norm"] = np.mean(
        residuals[[f"norm_{x}" for x in cams]].to_numpy(), axis=1
    )
    res_stas = residuals.describe()
    res_stas_s = res_stas.stack()

    if not Path(fname).exists():
        with open(cfg.residuals_fname, "w") as f:
            header_line = (
                "ep"
                + sep
                + f"{sep}".join([f"{x[0]}-{x[1]}" for x in res_stas_s.index.to_list()])
            )
            f.write(header_line + "\n")
    with open(fname, "a") as f:
        line = (
            epoch_dict[ep] + sep + f"{sep}".join([str(x) for x in res_stas_s.to_list()])
        )
        f.write(line + "\n")


def make_matching_plot(epoch, ep, out_dir, show_fig=False):
    import matplotlib
    from matplotlib import pyplot as plt

    from icepy4d.visualization import plot_features

    matplotlib.use("tkagg")
    cams = list(epoch.cameras.keys())
    features = epoch.features
    images = epoch.images

    fig, axes = plt.subplots(1, 2)
    titles = ["C1", "C2"]
    for cam, ax, title in zip(cams, axes, titles):
        plot_features(
            images[cam].value,
            features[cam],
            ax=ax,
            s=2,
            linewidths=0.3,
        )
        ax.set_title(f"{title}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(
        out_dir / f"matches_{epoch.datetime.strftime('%Y_%m_%d')}.png",
        dpi=300,
    )

    if show_fig:
        plt.show()
    else:
        plt.close()


""" Inizialize Variables """
if len(sys.argv) > 1:
    # If given, parse inputs from command line
    cfg_file, log_cfg = inizialization.parse_command_line()

    # Setup logger
    icepy4d_utils.setup_logger(
        log_cfg["log_folder"],
        log_cfg["log_name"],
        log_cfg["log_file_level"],
        log_cfg["log_console_level"],
    )
else:
    cfg_file = Path(CFG_FILE)
    icepy4d_utils.setup_logger(console_log_level="info", logfile_level="info")

# Parse configuration file
logging.info(f"Configuration file: {cfg_file.stem}")
cfg = inizialization.parse_yaml_cfg(cfg_file)
timer_global = icepy4d_utils.AverageTimer()
cams = cfg.cams

# Inizialize variables
inizializer = inizialization.Inizializer(cfg)
images = inizializer.init_image_ds()
epoch_dict = inizializer.init_epoch_dict()
features_old = inizializer.init_features()
epoches = Epoches(starting_epoch=cfg.proc.epoch_to_process[0])

""" Big Loop over epoches """

logging.info("------------------------------------------------------")
logging.info("Processing started:")
timer = icepy4d_utils.AverageTimer()
iter = 0  # necessary only for printing the number of processed iteration
for ep in cfg.proc.epoch_to_process:
    logging.info("------------------------------------------------------")
    logging.info(
        f"Processing epoch {ep} [{iter}/{cfg.proc.epoch_to_process[-1]-cfg.proc.epoch_to_process[0]}] - {epoch_dict[ep]}..."
    )
    iter += 1
    epochdir = Path(cfg.paths.results_dir) / epoch_dict[ep]
    match_dir = epochdir / "matching"

    # Load existing epcoh
    if LOAD_EXISTING_SOLUTION:
        path = f"{epochdir}/{epoch_dict[ep]}.pickle"
        logging.info(f"Loading epoch from {path}")
        epoch = Epoch.read_pickle(path, ignore_errors=True)
        if epoch is not None:
            epoches.add_epoch(epoch)
            logging.info("Epoch loaded.")

            # matches_fig_dir = "res/fig_for_paper/matches_fig"
            # make_matching_plot(epoch, epoch, matches_fig_dir, show_fig=False)

            del epoch
            continue
        else:
            logging.error("Unable to import epoch.")
    else:
        # Create new epoch
        epoch = inizializer.init_epoch(epoch_id=ep, epoch_dir=epochdir)
        epoches.add_epoch(epoch)

        # NOTE: Move this part of code to a notebook for an example of how to create a new epoch
        # im_epoch: icepy4d_classes.ImagesDict = {
        #     cam: icepy4d_classes.Image(images[cam].get_image_path(ep)) for cam in cams
        # }

        # # Load targets
        # target_paths = [
        #     cfg.georef.target_dir / (im_epoch[cam].stem + cfg.georef.target_file_ext)
        #     for cam in cams
        # ]
        # targ_ep = icepy4d_classes.Targets(
        #     im_file_path=target_paths,
        #     obj_file_path=cfg.georef.target_dir / cfg.georef.target_world_file,
        # )

        # # Load cameras
        # cams_ep: icepy4d_classes.CamerasDict = {}
        # for cam in cams:
        #     calib = icepy4d_classes.Calibration(
        #         cfg.paths.calibration_dir / f"{cam}.txt"
        #     )
        #     cams_ep[cam] = calib.to_camera()

        # cams_ep = {
        #     cam: icepy4d_classes.Calibration(
        #         cfg.paths.calibration_dir / f"{cam}.txt"
        #     ).to_camera()
        #     for cam in cams
        # }

        # # init empty features and points
        # feat_ep = {cam: icepy4d_classes.Features() for cam in cams}
        # pts_ep = icepy4d_classes.Points()

        # epoch = Epoch(
        #     im_epoch[cams[0]].datetime,
        #     images=im_epoch,
        #     cameras=cams_ep,
        #     features=feat_ep,
        #     points=pts_ep,
        #     targets=targ_ep,
        #     point_cloud=None,
        #     epoch_dir=epochdir,
        # )
        # epoches.add_epoch(epoch)

        # del im_epoch, cams_ep, feat_ep, pts_ep, targ_ep, target_paths

    # Perform matching and tracking
    if cfg.proc.do_matching:
        if DO_PRESELECTION:
            if cfg.proc.do_tracking and ep > cfg.proc.epoch_to_process[0]:
                epoch.features = tracking_base(
                    images,
                    epoches[ep - 1].features,
                    cams,
                    epoch_dict,
                    ep,
                    cfg.tracking,
                    epochdir,
                )

            epoch.features = match_by_preselection(
                images,
                epoch.features,
                cams,
                ep,
                cfg.matching,
                match_dir,
                n_tiles=4,
                n_dist=1.5,
                viz_results=True,
                fast_viz=True,
            )
        else:
            features_old = MatchingAndTracking(
                cfg=cfg,
                epoch=ep,
                images=images,
                features=features_old,
                epoch_dict=epoch_dict,
            )
            epoch.features = features_old[ep]

        # Run additional matching on selected patches:
        if DO_ADDITIONAL_MATCHING:
            logging.info("Performing additional matching on user-specified patches")
            im_stems = [epoch.images[cam].stem for cam in cams]
            sg_opt = {
                "weights": cfg.matching.weights,
                "keypoint_threshold": 0.0001,
                "max_keypoints": 8192,
                "match_threshold": 0.2,
                "force_cpu": False,
            }
            for i, patches_lim in enumerate(PATCHES):
                find_matches_on_patches(
                    images=images,
                    patches_lim=patches_lim,
                    epoch=ep,
                    features=epoch.features,
                    cfg=sg_opt,
                    do_geometric_verification=True,
                    geometric_verification_threshold=10,
                    viz_results=True,
                    fast_viz=True,
                    viz_path=match_dir
                    / f"{im_stems[0]}_{im_stems[1]}_matches_patch_{i}.png",
                )

            # Run again geometric verification
            geometric_verification(
                epoch.features,
                threshold=cfg.matching.pydegensac_threshold,
                confidence=cfg.matching.pydegensac_confidence,
            )
            logging.info("Matching by patches completed.")

            # For debugging
            # for cam in cams:
            #     epoch.features[cam].plot_features(images[cam].read_image(ep).value)
    else:
        try:
            epoch.features = load_matches_from_disk(match_dir)
        except FileNotFoundError as err:
            logging.exception(err)
            logging.warning("Performing new matching and tracking...")
            features_old = MatchingAndTracking(
                cfg=cfg,
                epoch=ep,
                images=images,
                features=features_old,
                epoch_dict=epoch_dict,
            )
            epoch.features = features_old[ep]

    timer.update("matching")

    """ SfM """

    logging.info(f"Reconstructing epoch {ep}...")

    # --- Space resection of Master camera ---#
    # At the first ep, perform Space resection of the first camera by using GCPs. At all other epoches, set camera 1 EO equal to first one.
    if cfg.proc.do_space_resection and ep == 0:
        """Initialize Single_camera_geometry class with a cameras object"""
        space_resection = abs_ori.Space_resection(epoch.cameras[cams[0]])
        space_resection.estimate(
            epoch.targets.get_image_coor_by_label(cfg.georef.targets_to_use, cam_id=0)[
                0
            ],
            epoch.targets.get_object_coor_by_label(cfg.georef.targets_to_use)[0],
        )
        # Store result in camera 0 object
        epoch.cameras[cams[0]] = space_resection.camera

    # --- Perform Relative orientation of the two cameras ---#
    # Initialize RelativeOrientation class with a list containing the two cameras and a list contaning the matched features location on each camera.
    # @TODO: decide wheter to do a deep copy of the arguments or directly modify them in the function (and state it in docs).
    relative_ori = sfm.RelativeOrientation(
        [epoch.cameras[cams[0]], epoch.cameras[cams[1]]],
        [
            epoch.features[cams[0]].kpts_to_numpy(),
            epoch.features[cams[1]].kpts_to_numpy(),
        ],
    )
    relative_ori.estimate_pose(
        threshold=cfg.matching.pydegensac_threshold,
        confidence=0.999999,
        scale_factor=np.linalg.norm(
            cfg.georef.camera_centers_world[0] - cfg.georef.camera_centers_world[1]
        ),
    )
    # Store result in camera 1 object
    epoch.cameras[cams[1]] = relative_ori.cameras[1]

    # --- Triangulate Points ---#
    # Initialize a Triangulate class instance with a list containing the two cameras and a list contaning the matched features location on each camera. Triangulated points are saved as points3d proprierty of the Triangulate object (eg., triangulation.points3d)
    triang = sfm.Triangulate(
        [epoch.cameras[cams[0]], epoch.cameras[cams[1]]],
        [
            epoch.features[cams[0]].kpts_to_numpy(),
            epoch.features[cams[1]].kpts_to_numpy(),
        ],
    )
    points3d = triang.triangulate_two_views(
        compute_colors=True, image=images[cams[1]].read_image(ep).value, cam_id=1
    )
    logging.info("Tie points triangulated.")

    # --- Absolute orientation (-> coregistration on stable points) ---#
    if cfg.proc.do_coregistration:
        # Get targets available in all cameras
        # Labels of valid targets are returned as second element by get_image_coor_by_label() method
        valid_targets = epoch.targets.get_image_coor_by_label(
            cfg.georef.targets_to_use, cam_id=0
        )[1]
        for id in range(1, len(cams)):
            assert (
                valid_targets
                == epoch.targets.get_image_coor_by_label(
                    cfg.georef.targets_to_use, cam_id=id
                )[1]
            ), f"epoch {ep} - {epoch_dict[ep]}: Different targets found in image {id} - {images[cams[id]][ep]}"
        if len(valid_targets) < 1:
            logging.error(
                f"Not enough targets found. Skipping epoch {ep} and moving to next epoch"
            )
            continue
        if valid_targets != cfg.georef.targets_to_use:
            logging.warning(f"Not all targets found. Using onlys {valid_targets}")

        image_coords = [
            epoch.targets.get_image_coor_by_label(valid_targets, cam_id=id)[0]
            for id, cam in enumerate(cams)
        ]
        obj_coords = epoch.targets.get_object_coor_by_label(valid_targets)[0]
        try:
            abs_ori = sfm.Absolute_orientation(
                (epoch.cameras[cams[0]], epoch.cameras[cams[1]]),
                points3d_final=obj_coords,
                image_points=image_coords,
                camera_centers_world=cfg.georef.camera_centers_world,
            )
            T = abs_ori.estimate_transformation_linear(estimate_scale=True)
            points3d = abs_ori.apply_transformation(points3d=points3d)
            for i, cam in enumerate(cams):
                epoch.cameras[cam] = abs_ori.cameras[i]
            logging.info("Absolute orientation completed.")
        except ValueError as err:
            logging.error(err)
            logging.error(
                f"Absolute orientation not succeded. Not enough targets available. Skipping epoch {ep} and moving to next epoch"
            )
            continue

    # Create point cloud and save .ply to disk
    # pcd_epc = icepy4d_classes.PointCloud(points3d=points3d, points_col=triang.colors)
    pts = icepy4d_classes.Points()
    pts.append_points_from_numpy(
        points3d,
        track_ids=epoch.features[cams[0]].get_track_ids(),
        colors=triang.colors,
    )

    timer.update("relative orientation")

    # Metashape BBA and dense cloud
    if cfg.proc.do_metashape_processing:
        # If a metashape folder is already present, delete it completely and start a new metashape project
        metashape_path = epochdir / "metashape"
        if metashape_path.exists() and cfg.metashape.force_overwrite_projects:
            logging.warning(
                f"Metashape folder {metashape_path} already exists, but force_overwrite_projects is set to True. Removing all old Metashape files"
            )
            shutil.rmtree(metashape_path, ignore_errors=True)

        # Export results in Bundler format
        im_dict = {cam: images[cam].get_image_path(ep) for cam in cams}
        write_bundler_out(
            export_dir=epochdir,
            im_dict=im_dict,
            cameras=epoch.cameras,
            features=epoch.features,
            points=pts,
            targets=epoch.targets,
            targets_to_use=valid_targets,
            targets_enabled=[True for el in valid_targets],
        )

        ms_cfg = MS.build_metashape_cfg(cfg, epoch_dict, ep)
        ms = MS.MetashapeProject(ms_cfg, timer)
        ms.run_full_workflow()

        ms_reader = MS.MetashapeReader(
            metashape_dir=epochdir / "metashape",
            num_cams=len(cams),
        )
        ms_reader.read_icepy4d_outputs()
        # for i, cam in enumerate(cams):
        #     focals[cam][ep] = ms_reader.get_focal_lengths()[i]

        # Assign camera extrinsics and intrinsics estimated in Metashape to Camera Object (assignation is done manaully @TODO automatic K and extrinsics matrixes to assign correct camera by camera label)
        new_K = ms_reader.get_K()
        epoch.cameras[cams[0]].update_K(new_K[1])
        epoch.cameras[cams[1]].update_K(new_K[0])

        epoch.cameras[cams[0]].update_extrinsics(
            ms_reader.extrinsics[images[cams[0]].get_image_stem(ep)]
        )
        epoch.cameras[cams[1]].update_extrinsics(
            ms_reader.extrinsics[images[cams[1]].get_image_stem(ep)]
        )

        # Triangulate again points and update Point Cloud dict
        triang = sfm.Triangulate(
            [epoch.cameras[cams[0]], epoch.cameras[cams[1]]],
            [
                epoch.features[cams[0]].kpts_to_numpy(),
                epoch.features[cams[1]].kpts_to_numpy(),
            ],
        )
        points3d = triang.triangulate_two_views(
            compute_colors=True,
            image=images[cams[1]].read_image(ep).value,
            cam_id=1,
        )

        # pcd_epc = icepy4d_classes.PointCloud(
        #     points3d=points3d, points_col=triang.colors
        # )

        epoch.points.append_points_from_numpy(
            points3d,
            track_ids=epoch.features[cams[0]].get_track_ids(),
            colors=triang.colors,
        )

        if cfg.proc.save_sparse_cloud:
            epoch.points.to_point_cloud().write_ply(
                cfg.paths.results_dir / f"point_clouds/sparse_{epoch_dict[ep]}.ply"
            )

        # - For debugging purposes
        # from icepy4d.visualization import plot_features
        # from matplotlib import pyplot as plt
        # import matplotlib
        # matplotlib.use("tkagg")

        # M = epoch.targets.get_object_coor_by_label(cfg.georef.targets_to_use)[0]
        # m = epoch.cameras[cams[1]].project_point(M)
        # plot_features(images[cams[1]].read_image(ep).value, m)
        # plot_features(
        #     images[cams[0]].read_image(ep).value,
        #     epoch.features[cams[0]].kpts_to_numpy(),
        # )

        # cam = cams[0]
        # f0 = epoch.features[cam]
        # plot_features(images[cam].read_image(ep).value, f0)
        # plt.show()

        # Clean variables
        del relative_ori, triang, abs_ori, points3d
        del T, new_K
        del ms_cfg, ms, ms_reader
        gc.collect()

        # Save epoch as a pickle object
        epoches[ep].save_pickle(f"{epochdir}/{epoch_dict[ep]}.pickle")

        # Save matches plot
        matches_fig_dir = "res/fig_for_paper/matches_fig"
        make_matching_plot(epoches[ep], ep, matches_fig_dir, show_fig=False)

        # Compute reprojection error
        compute_reprojection_error(cfg.residuals_fname, epoches[ep])

        # Save focal length to file
        write_cameras_to_disk(cfg.camea_estimated_fname, epoches[ep], epoch_dict[ep])

    timer.print(f"Epoch {ep} completed")

timer_global.update("ICEpy4D processing")


# Homograpghy warping
if cfg.proc.do_homography_warping:
    from copy import deepcopy

    from icepy4d.thirdparty.transformations import euler_from_matrix, euler_matrix
    from icepy4d.utils.homography import homography_warping

    logging.info("Performing homograpy warping for DIC")

    reference_day = "2022_07_28"
    do_smoothing = True
    use_median = True

    # reference_epoch = list(epoch_dict.values()).index(reference_day)
    cam = cfg.proc.camera_to_warp
    ref_epoch = epoches.get_epoch_by_date(reference_day)
    cam_ref = ref_epoch.cameras[cam]

    for ep in cfg.proc.epoch_to_process:
        # Camera pose smoothing
        if do_smoothing:
            match str(ep):
                case "0":
                    epoch_range = range(ep + 0, ep + 5)
                case "1":
                    epoch_range = range(ep - 1, ep + 4)
                case "158":
                    epoch_range = range(ep - 3, ep + 2)
                case "159":
                    epoch_range = range(ep - 4, ep + 1)
                case other:
                    epoch_range = range(ep - 2, ep + 3)
            cam_to_warp = deepcopy(epoch.cameras[cam])
            angles = np.stack(
                [euler_from_matrix(epoches.get_epoch_id(e).cameras[cam].R) for e in epoch_range], axis=1
            )
            
            if use_median:
                ang = np.median(angles, axis=1)
            else:
                ang = np.mean(angles, axis=1)
            extrinsics_med = deepcopy(cam_to_warp.extrinsics)
            extrinsics_med[:3, :3] = euler_matrix(*ang)[:3, :3]
            cam_to_warp.update_extrinsics(extrinsics_med)
        else:
            cam_to_warp = epoch.cameras[cam]

        _ = homography_warping(
            cam_0=cam_ref,
            cam_1=cam_to_warp,
            image=images[cam].read_image(ep).value,
            undistort=True,
            out_path=f"res/warped/{images[cam][ep]}",
        )

    timer_global.update("Homograpy warping")

timer_global.print("Total time elapsed")

logging.info("Processing completed.")
