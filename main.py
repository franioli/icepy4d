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
import sys
import logging
import shutil
from pathlib import Path

import numpy as np

# icepy4d4D
import icepy4d.classes as icepy4d_classes
import icepy4d.metashape.metashape as MS
import icepy4d.sfm as sfm
import icepy4d.utils as icepy4d_utils
import icepy4d.utils.initialization as inizialization
import icepy4d.visualization as icepy4d_viz
from icepy4d.classes.solution import Solution
from icepy4d.io.export2bundler import write_bundler_out
from icepy4d.matching.match_by_preselection import (
    find_matches_on_patches,
    match_by_preselection,
)
from icepy4d.matching.matching_base import MatchingAndTracking
from icepy4d.matching.tracking_base import tracking_base
from icepy4d.matching.utils import geometric_verification, load_matches_from_disk

# Temporary parameters TODO: put them in config file
LOAD_EXISTING_SOLUTION = False  # False #
DO_PRESELECTION = False
DO_ADDITIONAL_MATCHING = True
PATCHES = [
    {"p1": [0, 500, 2000, 2000], "p2": [4000, 0, 6000, 1500]},
    {"p1": [1000, 1500, 4500, 2500], "p2": [1500, 1500, 5000, 2500]},
    {"p1": [2000, 2000, 3000, 3000], "p2": [2100, 2100, 3100, 3100]},
    {"p1": [2300, 1700, 3300, 2700], "p2": [3000, 1900, 4000, 2900]},
]


def write_cameras_to_disk(fname, solution, date):
    from icepy4d.thirdparty.transformations import euler_from_matrix

    if solution is None:
        return
    with open(fname, "a") as file:
        file.write(f"{date}")
        for cam in solution.cameras.keys():
            f = solution.cameras[cam].K[1, 1]
            R = solution.cameras[cam].R
            o, p, k = euler_from_matrix(R)
            o, p, k = np.rad2deg(o), np.rad2deg(p), np.rad2deg(k)
            file.write(f",{f:.2f},{o:.4f},{p:.4f},{k:.4f}")
        file.write("\n")


focals_fname = cfg.paths.results_dir / "focal_length.txt"
with open(focals_fname, "w") as file:
    file.write(f"date,f1,omega1,phi1,kappa1,f2,omega2,phi2,kappa2\n")


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
    cfg_file = Path("config/config_2022_exp.yaml")
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

""" Big Loop over epoches """

logging.info("------------------------------------------------------")
logging.info("Processing started:")
timer = icepy4d_utils.AverageTimer()
iter = 0  # necessary only for printing the number of processed iteration
for epoch in cfg.proc.epoch_to_process:
    logging.info("------------------------------------------------------")
    logging.info(
        f"Processing epoch {epoch} [{iter}/{cfg.proc.epoch_to_process[-1]-cfg.proc.epoch_to_process[0]}] - {epoch_dict[epoch]}..."
    )
    iter += 1

    epochdir = Path(cfg.paths.results_dir) / epoch_dict[epoch]
    match_dir = epochdir / "matching"

    if LOAD_EXISTING_SOLUTION:
        path = f"{epochdir}/{epoch_dict[epoch]}.pickle"
        logging.info(f"Loading solution from {path}")
        solution = Solution.read_solution(path, ignore_errors=True)
        if solution is not None:
            cameras[epoch], _, features[epoch], points[epoch] = solution
            del solution
            logging.info("Solution loaded.")
            continue
        else:
            logging.error("Unable to import solution.")

    # Perform matching and tracking
    if cfg.proc.do_matching:
        if DO_PRESELECTION:
            if cfg.proc.do_tracking and epoch > cfg.proc.epoch_to_process[0]:
                features[epoch] = tracking_base(
                    images,
                    features[epoch - 1],
                    cams,
                    epoch_dict,
                    epoch,
                    cfg.tracking,
                    epochdir,
                )

            features[epoch] = match_by_preselection(
                images,
                features[epoch],
                cams,
                epoch,
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
                epoch=epoch,
                images=images,
                features=features,
                epoch_dict=epoch_dict,
            )

        # Run additional matching on selected patches:
        if DO_ADDITIONAL_MATCHING:
            logging.info("Performing additional matching on user-specified patches")
            im_stems = [images[cam].get_image_stem(epoch) for cam in cams]
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
                    epoch=epoch,
                    features=features[epoch],
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
                features[epoch],
                threshold=cfg.matching.pydegensac_threshold,
                confidence=cfg.matching.pydegensac_confidence,
            )
            logging.info("Matching by patches completed.")

            # For debugging
            # for cam in cams:
            #     features[epoch][cam].plot_features(images[cam].read_image(epoch).value)
    else:
        try:
            features[epoch] = load_matches_from_disk(match_dir)
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

    """ SfM """

    logging.info(f"Reconstructing epoch {epoch}...")

    # --- Space resection of Master camera ---#
    # At the first epoch, perform Space resection of the first camera by using GCPs. At all other epoches, set camera 1 EO equal to first one.
    if cfg.proc.do_space_resection and epoch == 0:
        """Initialize Single_camera_geometry class with a cameras object"""
        space_resection = abs_ori.Space_resection(cameras[epoch][cams[0]])
        space_resection.estimate(
            targets[epoch].get_image_coor_by_label(cfg.georef.targets_to_use, cam_id=0)[
                0
            ],
            targets[epoch].get_object_coor_by_label(cfg.georef.targets_to_use)[0],
        )
        # Store result in camera 0 object
        cameras[epoch][cams[0]] = space_resection.camera

    # --- Perform Relative orientation of the two cameras ---#
    # Initialize RelativeOrientation class with a list containing the two cameras and a list contaning the matched features location on each camera.
    # @TODO: decide wheter to do a deep copy of the arguments or directly modify them in the function (and state it in docs).
    relative_ori = sfm.RelativeOrientation(
        [cameras[epoch][cams[0]], cameras[epoch][cams[1]]],
        [
            features[epoch][cams[0]].kpts_to_numpy(),
            features[epoch][cams[1]].kpts_to_numpy(),
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
    # pcd_epc = icepy4d_classes.PointCloud(points3d=points3d, points_col=triang.colors)
    pts = icepy4d_classes.Points()
    pts.append_points_from_numpy(
        points3d,
        track_ids=features[epoch][cams[0]].get_track_ids(),
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
        im_dict = {cam: images[cam].get_image_path(epoch) for cam in cams}
        write_bundler_out(
            export_dir=epochdir,
            im_dict=im_dict,
            cameras=cameras[epoch],
            features=features[epoch],
            points=pts,
            targets=targets[epoch],
            targets_to_use=valid_targets,
            targets_enabled=[True for el in valid_targets],
        )

        ms_cfg = MS.build_metashape_cfg(cfg, epoch_dict, epoch)
        ms = MS.MetashapeProject(ms_cfg, timer)
        ms.run_full_workflow()

        ms_reader = MS.MetashapeReader(
            metashape_dir=epochdir / "metashape",
            num_cams=len(cams),
        )
        ms_reader.read_icepy4d_outputs()
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

        # pcd_epc = icepy4d_classes.PointCloud(
        #     points3d=points3d, points_col=triang.colors
        # )

        points[epoch].append_points_from_numpy(
            points3d,
            track_ids=features[epoch][cams[0]].get_track_ids(),
            colors=triang.colors,
        )

        if cfg.proc.save_sparse_cloud:
            points[epoch].to_point_cloud().write_ply(
                cfg.paths.results_dir / f"point_clouds/sparse_{epoch_dict[epoch]}.ply"
            )

        # - For debugging purposes
        # M = targets[epoch].get_object_coor_by_label(cfg.georef.targets_to_use)[0]
        # m = cameras[epoch][cams[1]].project_point(M)
        # plot_features(images[cams[1]].read_image(epoch).value, m)
        # plot_features(images[cams[0]].read_image(epoch).value, features[epoch][cams[0]].kpts_to_numpy())

        # Clean variables
        del relative_ori, triang, abs_ori, points3d
        del T, new_K
        del ms_cfg, ms, ms_reader
        gc.collect()

        # Homograpghy warping
        if cfg.proc.do_homography_warping:
            ep_ini = cfg.proc.epoch_to_process[0]
            cam = cfg.proc.camera_to_warp
            image = images[cams[1]].read_image(epoch).value
            out_path = f"res/warped/{images[cam][epoch]}"
            icepy4d_utils.homography_warping(
                cameras[ep_ini][cam], cameras[epoch][cam], image, out_path, timer
            )

        # Save solution as a pickle object
        solution = Solution(cameras[epoch], images, features[epoch], points[epoch])
        solution.save_solutions(f"{epochdir}/{epoch_dict[epoch]}.pickle")

        # Save focal length to file
        write_cameras_to_disk(focals_fname, solution, epoch_dict[epoch])

        del solution

    timer.print(f"Epoch {epoch} completed")

timer_global.update("SfM")

logging.info("Processing completed.")
