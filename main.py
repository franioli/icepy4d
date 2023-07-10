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
LOAD_EXISTING_SOLUTION = False  # False #
DO_PRESELECTION = False
DO_ADDITIONAL_MATCHING = True
PATCHES = [
    {"p1": [0, 500, 2000, 2000], "p2": [4000, 0, 6000, 1500]},
    {"p1": [1000, 1500, 4500, 2500], "p2": [1500, 1500, 5000, 2500]},
    {"p1": [2000, 2000, 3000, 3000], "p2": [2100, 2100, 3100, 3100]},
    {"p1": [2300, 1700, 3300, 2700], "p2": [3000, 1900, 4000, 2900]},
]


def write_cameras_to_disk(fname, solution, date, sep=","):
    from icepy4d.thirdparty.transformations import euler_from_matrix

    if solution is None:
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
        with open(camea_estimated_fname, "w") as file:
            file.write(f"{f'{sep}'.join(items)}\n")

    with open(fname, "a") as file:
        file.write(f"{date}")
        for cam in solution.cameras.keys():
            f = solution.cameras[cam].K[1, 1]
            # R = solution.cameras[cam].R
            R = solution.cameras[cam].pose[:3, :3]
            o, p, k = euler_from_matrix(R)
            o, p, k = np.rad2deg(o), np.rad2deg(p), np.rad2deg(k)
            file.write(f"{sep}{f:.2f}{sep}{o:.4f}{sep}{p:.4f}{sep}{k:.4f}")
        file.write("\n")


def compute_reprojection_error(fname, solution, sep=","):
    print("Computing reprojection error")

    import pandas as pd

    residuals = pd.DataFrame()
    for cam_key, camera in solution.cameras.items():
        feat = solution.features[cam_key]
        projections = camera.project_point(solution.points.to_numpy())
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
        with open(residuals_fname, "w") as f:
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


def make_matching_plot(solution, ep, out_dir, show_fig=False):
    import matplotlib
    from matplotlib import pyplot as plt

    from icepy4d.visualization import plot_features

    matplotlib.use("tkagg")
    cams = list(solution.cameras.keys())
    features = solution.features
    images = solution.images

    fig, axes = plt.subplots(1, 2)
    titles = ["C1", "C2"]
    for cam, ax, title in zip(cams, axes, titles):
        plot_features(
            images[cam].read_image(ep).value,
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
        out_dir / f"matches_{epoch_dict[ep]}.png",
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
    icepy4d_utils.setup_logger()

# Parse configuration file
logging.info(f"Configuration file: {cfg_file.stem}")
cfg = inizialization.parse_yaml_cfg(cfg_file)

timer_global = icepy4d_utils.AverageTimer()

inizializer = inizialization.Inizializer(cfg)
inizializer.inizialize_icepy4d()
cams = inizializer.cams
images = inizializer.images
epoch_dict = inizializer.epoch_dict
cameras = inizializer.cameras
features = inizializer.features
targets = inizializer.targets
points = inizializer.points
focals = inizializer.focals_dict

# TEMPORARY: initialize file for saving output for paper
# These outputs must be formalized for the final version
camea_estimated_fname = cfg.paths.results_dir / "camera_info_est.txt"
residuals_fname = cfg.paths.results_dir / "residuals_image.txt"
matching_stats_fname = cfg.paths.results_dir / "matching_tracking_results.txt"

# remove files if they already exist
if camea_estimated_fname.exists():
    camea_estimated_fname.unlink()
if residuals_fname.exists():
    residuals_fname.unlink()
if matching_stats_fname.exists():
    matching_stats_fname.unlink()


solutions = {}


epoches = Epoches()




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

    # Create epoch
    epoch = Epoch()

    match_dir = epochdir / "matching"

    if LOAD_EXISTING_SOLUTION:
        path = f"{epochdir}/{epoch_dict[ep]}.pickle"
        logging.info(f"Loading solution from {path}")
        solution = Epoch.read_pickle(path, ignore_errors=True)
        if solution is not None:
            solutions[ep] = solution
            cameras[ep], _, features[ep], points[ep] = solution
            logging.info("Epoch loaded.")

            # matches_fig_dir = "res/fig_for_paper/matches_fig"
            # make_matching_plot(solution, epoch, matches_fig_dir, show_fig=False)

            del solution
            continue
        else:
            logging.error("Unable to import solution.")

    # Perform matching and tracking
    if cfg.proc.do_matching:
        if DO_PRESELECTION:
            if cfg.proc.do_tracking and ep > cfg.proc.epoch_to_process[0]:
                features[ep] = tracking_base(
                    images,
                    features[ep - 1],
                    cams,
                    epoch_dict,
                    ep,
                    cfg.tracking,
                    epochdir,
                )

            features[ep] = match_by_preselection(
                images,
                features[ep],
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
            features = MatchingAndTracking(
                cfg=cfg,
                epoch=ep,
                images=images,
                features=features,
                epoch_dict=epoch_dict,
            )

        # Run additional matching on selected patches:
        if DO_ADDITIONAL_MATCHING:
            logging.info("Performing additional matching on user-specified patches")
            im_stems = [images[cam].get_image_stem(ep) for cam in cams]
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
                    features=features[ep],
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
                features[ep],
                threshold=cfg.matching.pydegensac_threshold,
                confidence=cfg.matching.pydegensac_confidence,
            )
            logging.info("Matching by patches completed.")

            # For debugging
            # for cam in cams:
            #     features[ep][cam].plot_features(images[cam].read_image(ep).value)
    else:
        try:
            features[ep] = load_matches_from_disk(match_dir)
        except FileNotFoundError as err:
            logging.exception(err)
            logging.warning("Performing new matching and tracking...")
            features = MatchingAndTracking(
                cfg=cfg,
                epoch=ep,
                images=images,
                features=features,
                epoch_dict=epoch_dict,
            )

    timer.update("matching")

    """ SfM """

    logging.info(f"Reconstructing epoch {ep}...")

    # --- Space resection of Master camera ---#
    # At the first ep, perform Space resection of the first camera by using GCPs. At all other epoches, set camera 1 EO equal to first one.
    if cfg.proc.do_space_resection and ep == 0:
        """Initialize Single_camera_geometry class with a cameras object"""
        space_resection = abs_ori.Space_resection(cameras[ep][cams[0]])
        space_resection.estimate(
            targets[ep].get_image_coor_by_label(cfg.georef.targets_to_use, cam_id=0)[0],
            targets[ep].get_object_coor_by_label(cfg.georef.targets_to_use)[0],
        )
        # Store result in camera 0 object
        cameras[ep][cams[0]] = space_resection.camera

    # --- Perform Relative orientation of the two cameras ---#
    # Initialize RelativeOrientation class with a list containing the two cameras and a list contaning the matched features location on each camera.
    # @TODO: decide wheter to do a deep copy of the arguments or directly modify them in the function (and state it in docs).
    relative_ori = sfm.RelativeOrientation(
        [cameras[ep][cams[0]], cameras[ep][cams[1]]],
        [
            features[ep][cams[0]].kpts_to_numpy(),
            features[ep][cams[1]].kpts_to_numpy(),
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
    cameras[ep][cams[1]] = relative_ori.cameras[1]

    # --- Triangulate Points ---#
    # Initialize a Triangulate class instance with a list containing the two cameras and a list contaning the matched features location on each camera. Triangulated points are saved as points3d proprierty of the Triangulate object (eg., triangulation.points3d)
    triang = sfm.Triangulate(
        [cameras[ep][cams[0]], cameras[ep][cams[1]]],
        [
            features[ep][cams[0]].kpts_to_numpy(),
            features[ep][cams[1]].kpts_to_numpy(),
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
        valid_targets = targets[ep].get_image_coor_by_label(
            cfg.georef.targets_to_use, cam_id=0
        )[1]
        for id in range(1, len(cams)):
            assert (
                valid_targets
                == targets[ep].get_image_coor_by_label(
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
            targets[ep].get_image_coor_by_label(valid_targets, cam_id=id)[0]
            for id, cam in enumerate(cams)
        ]
        obj_coords = targets[ep].get_object_coor_by_label(valid_targets)[0]
        try:
            abs_ori = sfm.Absolute_orientation(
                (cameras[ep][cams[0]], cameras[ep][cams[1]]),
                points3d_final=obj_coords,
                image_points=image_coords,
                camera_centers_world=cfg.georef.camera_centers_world,
            )
            T = abs_ori.estimate_transformation_linear(estimate_scale=True)
            points3d = abs_ori.apply_transformation(points3d=points3d)
            for i, cam in enumerate(cams):
                cameras[ep][cam] = abs_ori.cameras[i]
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
        track_ids=features[ep][cams[0]].get_track_ids(),
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
            cameras=cameras[ep],
            features=features[ep],
            points=pts,
            targets=targets[ep],
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
        for i, cam in enumerate(cams):
            focals[cam][ep] = ms_reader.get_focal_lengths()[i]

        # Assign camera extrinsics and intrinsics estimated in Metashape to Camera Object (assignation is done manaully @TODO automatic K and extrinsics matrixes to assign correct camera by camera label)
        new_K = ms_reader.get_K()
        cameras[ep][cams[0]].update_K(new_K[1])
        cameras[ep][cams[1]].update_K(new_K[0])

        cameras[ep][cams[0]].update_extrinsics(
            ms_reader.extrinsics[images[cams[0]].get_image_stem(ep)]
        )
        cameras[ep][cams[1]].update_extrinsics(
            ms_reader.extrinsics[images[cams[1]].get_image_stem(ep)]
        )

        # Triangulate again points and update Point Cloud dict
        triang = sfm.Triangulate(
            [cameras[ep][cams[0]], cameras[ep][cams[1]]],
            [
                features[ep][cams[0]].kpts_to_numpy(),
                features[ep][cams[1]].kpts_to_numpy(),
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

        points[ep].append_points_from_numpy(
            points3d,
            track_ids=features[ep][cams[0]].get_track_ids(),
            colors=triang.colors,
        )

        if cfg.proc.save_sparse_cloud:
            points[ep].to_point_cloud().write_ply(
                cfg.paths.results_dir / f"point_clouds/sparse_{epoch_dict[ep]}.ply"
            )

        # - For debugging purposes
        # from icepy4d.visualization import plot_features
        # from matplotlib import pyplot as plt
        # import matplotlib
        # matplotlib.use("tkagg")

        # M = targets[ep].get_object_coor_by_label(cfg.georef.targets_to_use)[0]
        # m = cameras[ep][cams[1]].project_point(M)
        # plot_features(images[cams[1]].read_image(ep).value, m)
        # plot_features(
        #     images[cams[0]].read_image(ep).value,
        #     features[ep][cams[0]].kpts_to_numpy(),
        # )

        # cam = cams[0]
        # f0 = features[ep][cam]
        # plot_features(images[cam].read_image(ep).value, f0)
        # plt.show()

        # Clean variables
        del relative_ori, triang, abs_ori, points3d
        del T, new_K
        del ms_cfg, ms, ms_reader
        gc.collect()

        # Save solution as a pickle object
        solutions[ep] = Epoch(
            datetime=datetime.strptime(epoch_dict[ep], "%Y_%m_%d"),
            epoch_id=ep,
            cameras=cameras[ep],
            images=images,
            features=features[ep],
            points=points[ep],
        )
        solutions[ep].save_pickle(f"{epochdir}/{epoch_dict[ep]}.pickle")

        # Save matches plot
        matches_fig_dir = "res/fig_for_paper/matches_fig"
        make_matching_plot(solutions[ep], ep, matches_fig_dir, show_fig=False)

        # Compute reprojection error
        compute_reprojection_error(residuals_fname, solutions[ep])

        # Save focal length to file
        write_cameras_to_disk(camea_estimated_fname, solutions[ep], epoch_dict[ep])

    timer.print(f"ep {ep} completed")

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

    reference_epoch = list(epoch_dict.values()).index(reference_day)
    cam = cfg.proc.camera_to_warp
    cam_ref = cameras[reference_epoch][cam]

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
            cam_to_warp = deepcopy(cameras[ep][cam])
            angles = np.stack(
                [euler_from_matrix(cameras[e][cam].R) for e in epoch_range], axis=1
            )
            if use_median:
                ang = np.median(angles, axis=1)
            else:
                ang = np.mean(angles, axis=1)
            extrinsics_med = deepcopy(cam_to_warp.extrinsics)
            extrinsics_med[:3, :3] = euler_matrix(*ang)[:3, :3]
            cam_to_warp.update_extrinsics(extrinsics_med)
        else:
            cam_to_warp = cameras[ep][cam]

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
