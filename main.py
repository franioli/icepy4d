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
from pathlib import Path

import numpy as np

# icepy4d4D
from icepy4d import classes as icepy4d_classes
from icepy4d import matching
from icepy4d import sfm
from icepy4d import io
from icepy4d import utils as icepy4d_utils
from icepy4d.classes.epoch import Epoch, Epoches
from icepy4d.metashape import metashape as MS
from icepy4d.utils import initialization as inizialization

# Temporary parameters TODO: put them in config file
CFG_FILE = "config/config_2022.yaml"
LOAD_EXISTING_SOLUTION = False
# DO_ADDITIONAL_MATCHING = False
# PATCHES = [
#     {"p1": [0, 500, 2000, 2000], "p2": [4000, 0, 6000, 1500]},
#     {"p1": [1000, 1500, 4500, 2500], "p2": [1500, 1500, 5000, 2500]},
#     {"p1": [2000, 2000, 3000, 3000], "p2": [2100, 2100, 3100, 3100]},
#     {"p1": [2300, 1700, 3300, 2700], "p2": [3000, 1900, 4000, 2900]},
# ]
# TODO: parse_yaml_cfg set deafults paths to results file, check this.


def make_matching_plot(epoch, out_dir, show_fig=False):
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
        out_dir / f"matches_{epoch.timestamp.strftime('%Y_%m_%d')}.png",
        dpi=300,
    )

    if show_fig:
        plt.show()
    else:
        plt.close()


def save_to_colmap():
    pass

    # Save features in colmap db
    print("AAAAA")
    import h5py
    from collections import defaultdict
    import torch
    from copy import deepcopy
    import os
    from icepy4d.io.colmap_utils.h5_to_db import (
        add_keypoints,
        add_matches,
        COLMAPDatabase,
    )
    import pycolmap

    MIN_MATCHES = 15
    colmap_dir = Path("colmap")
    colmap_dir.mkdir(exist_ok=True, parents=True)

    sg_features_fname = "features_sg.h5"
    with h5py.File(colmap_dir / sg_features_fname, mode="w") as f_match:
        key1 = epoch.images[cams[0]].name
        key2 = epoch.images[cams[1]].name
        group = f_match.require_group(key1)
        n_matches = len(epoch.features[cams[0]])

        mkpts0 = epoch.features[cams[0]].kpts_to_numpy()
        mkpts1 = epoch.features[cams[1]].kpts_to_numpy()
        if n_matches >= MIN_MATCHES:
            group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))

    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)
    with h5py.File(colmap_dir / sg_features_fname, mode="r") as f_match:
        for k1 in f_match.keys():
            group = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0] += total_kpts[k1]
                current_match[:, 1] += total_kpts[k2]
                total_kpts[k1] += len(matches)
                total_kpts[k2] += len(matches)
                match_indexes[k1][k2] = current_match
    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(
            torch.from_numpy(kpts[k]), dim=0, return_inverse=True
        )
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
            out_match[k1][k2] = m2.numpy()

    with h5py.File(colmap_dir / "keypoints.h5", mode="w") as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1

    with h5py.File(colmap_dir / "matches.h5", mode="w") as f_match:
        for k1, gr in out_match.items():
            group = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match

    # Create fake dir for colmap with symlinks
    img_dir = colmap_dir / "images"
    img_dir.mkdir(exist_ok=True, parents=True)
    for cam in cams:
        dst = img_dir / epoch.images[cam].name
        if not dst.exists():
            os.symlink(epoch.images[cam].path, dst)

    database_path = colmap_dir / "colmap.db"
    database_path.unlink(missing_ok=True)
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, colmap_dir, img_dir, "simple-radial", single_camera)
    add_matches(
        db,
        colmap_dir,
        fname_to_id,
    )
    db.commit()

    output_path = colmap_dir / "sparse"
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, colmap_dir, output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    maps[0].write(output_path)


""" Inizialize Variables """
if len(sys.argv) > 1:
    # If given, parse inputs from command line and setup logger
    cfg_file, log_cfg = inizialization.parse_command_line()
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

# Inizialize variables
cams = cfg.cams
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
        f"""Processing epoch {ep} [{iter}/{cfg.proc.epoch_to_process[-1]-cfg.proc.epoch_to_process[0]}] - {epoch_dict[ep]}..."""  # noqa: E501
    )
    iter += 1
    epochdir = cfg.paths.results_dir / epoch_dict[ep]
    match_dir = epochdir / "matching"

    # Load existing epcoh
    if LOAD_EXISTING_SOLUTION:
        path = epochdir / f"{epoch_dict[ep]}.pickle"
        epoch = Epoch.read_pickle(path)

        # For backward compatibility.
        # TODO: remove this when all epochs are saved with new format
        epoch._timestamp = epoch._datetime
        del epoch._datetime
        epoches.add_epoch(epoch)

        del epoch
        continue
    else:
        # Create new epoch
        epoch = inizializer.init_epoch(epoch_id=ep, epoch_dir=epochdir)
        epoches.add_epoch(epoch)

    # --- Matching and Tracking ---#
    # features_old = MatchingAndTracking(
    #     cfg=cfg,
    #     epoch=ep,
    #     images=images,
    #     features=features_old,
    #     epoch_dict=epoch_dict,
    # )
    # epoch.features = features_old[ep]

    matcher = matching.SuperGlueMatcher(cfg.matching)
    grid = [4, 3]
    overlap = 200
    matcher.match(
        epoch.images[cams[0]].value,
        epoch.images[cams[1]].value,
        quality=matching.Quality.HIGH,
        tile_selection=matching.TileSelection.PRESELECTION,
        grid=grid,
        overlap=overlap,
        do_viz_matches=True,
        do_viz_tiles=True,
        save_dir=match_dir,
        geometric_verification=matching.GeometricVerification.PYDEGENSAC,
        threshold=1,
        confidence=0.9999,
    )

    # TODO: implement this as a method of Matcher class
    from icepy4d.classes import Features

    f = {cam: Features() for cam in cams}
    f[cams[0]].append_features_from_numpy(
        x=matcher.mkpts0[:, 0],
        y=matcher.mkpts0[:, 1],
        descr=matcher.descriptors0,
        scores=matcher.scores0,
    )
    f[cams[1]].append_features_from_numpy(
        x=matcher.mkpts1[:, 0],
        y=matcher.mkpts1[:, 1],
        descr=matcher.descriptors1,
        scores=matcher.scores1,
    )
    epoch.features = f

    # # Run additional matching on selected patches:
    # if DO_ADDITIONAL_MATCHING:
    #     logging.info("Performing additional matching on user-specified patches")
    #     im_stems = [epoch.images[cam].stem for cam in cams]
    #     sg_opt = {
    #         "weights": cfg.matching.weights,
    #         "keypoint_threshold": 0.0001,
    #         "max_keypoints": 8192,
    #         "match_threshold": 0.2,
    #         "force_cpu": False,
    #     }
    #     for i, patches_lim in enumerate(PATCHES):
    #         find_matches_on_patches(
    #             images=images,
    #             patches_lim=patches_lim,
    #             epoch=ep,
    #             features=epoch.features,
    #             cfg=sg_opt,
    #             do_geometric_verification=True,
    #             geometric_verification_threshold=10,
    #             viz_results=True,
    #             fast_viz=True,
    #             viz_path=match_dir
    #             / f"{im_stems[0]}_{im_stems[1]}_matches_patch_{i}.png",
    #         )

    #     # Run again geometric verification
    #     geometric_verification(
    #         epoch.features,
    #         threshold=cfg.matching.pydegensac_threshold,
    #         confidence=cfg.matching.pydegensac_confidence,
    #     )
    #     logging.info("Matching by patches completed.")

    timer.update("matching")

    """ SfM """

    logging.info(f"Reconstructing epoch {ep}...")

    # --- Perform Relative orientation of the two cameras ---#
    # Initialize RelativeOrientation class with a list containing the two
    # cameras and a list contaning the matched features location on each camera.
    # @TODO: decide wheter to do a deep copy of the arguments or directly
    # modify them in the function (and state it in docs).
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
    # Initialize a Triangulate class instance with a list containing the two
    # cameras and a list contaning the matched features location on each
    # camera. Triangulated points are saved as points3d proprierty of the
    # Triangulate object (eg., triangulation.points3d)
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
        # Labels of valid targets are returned as second element by
        # get_image_coor_by_label() method
        valid_targets = epoch.targets.get_image_coor_by_label(
            cfg.georef.targets_to_use, cam_id=0
        )[1]
        for id in range(1, len(cams)):
            assert (
                valid_targets
                == epoch.targets.get_image_coor_by_label(
                    cfg.georef.targets_to_use, cam_id=id
                )[1]
            ), f"""epoch {ep} - {epoch_dict[ep]}: 
            Different targets found in image {id} - {images[cams[id]][ep]}"""
        if len(valid_targets) < 1:
            logging.error(
                f"Not enough targets found. Skipping epoch {ep} and moving to next epoch"  # noqa: E501
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
                f"""Absolute orientation failed. 
                Not enough targets available.
                Skipping epoch {ep} and moving to next epoch"""
            )
            continue

    # save_to_colmap()

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
        # If a metashape folder is already present,
        # delete it completely and start a new metashape project
        metashape_path = epochdir / "metashape"
        if metashape_path.exists() and cfg.metashape.force_overwrite_projects:
            logging.warning(
                f"""Metashape folder {metashape_path} already exists,
                but force_overwrite_projects is set to True.
                Removing all old Metashape files"""
            )
            shutil.rmtree(metashape_path, ignore_errors=True)

        # Export results in Bundler format
        im_dict = {cam: images[cam].get_image_path(ep) for cam in cams}
        io.write_bundler_out(
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

        # Assign camera extrinsics and intrinsics estimated in Metashape to
        # Camera Object (assignation is done manaully @TODO automatic K and
        # extrinsics matrixes to assign correct camera by camera label)
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
        # make_matching_plot(epoches[ep], ep, matches_fig_dir, show_fig=False)

        # Compute reprojection error
        io.write_reprojection_error_to_file(cfg.residuals_fname, epoches[ep])

        # Save focal length to file
        io.write_cameras_to_file(cfg.camera_estimated_fname, epoches[ep])

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
                [
                    euler_from_matrix(epoches.get_epoch_id(e).cameras[cam].R)
                    for e in epoch_range
                ],
                axis=1,
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
