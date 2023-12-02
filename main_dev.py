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
import shutil
from pathlib import Path

import numpy as np

# icepy4d4D
from icepy4d import core as icecore
from icepy4d.core import Epoch, Epoches, EpochDataMap
from icepy4d import matching
from icepy4d import sfm
from icepy4d import io
from icepy4d import utils
from icepy4d.metashape import metashape as MS
from icepy4d.utils import initialization

# Define configuration file
CFG_FILE = "config/config_2022.yaml"

# Parse configuration file
cfg = initialization.parse_cfg(CFG_FILE)
timer_global = utils.AverageTimer()
logger = utils.get_logger()

# initialize variables
epoch_map = EpochDataMap(cfg.paths.image_dir, time_tolerance_sec=1200)
epoches = Epoches(starting_epoch=cfg.proc.epoch_to_process[0])
cams = cfg.cams

""" Big Loop over epoches """

logger.info("------------------------------------------------------")
logger.info("Processing started:")
timer = utils.AverageTimer()
iter = 0  # necessary only for printing the number of processed iteration
for ep in cfg.proc.epoch_to_process:
    logger.info("------------------------------------------------------")
    logger.info(
        f"""Processing epoch {ep} [{iter}/{cfg.proc.epoch_to_process[-1]-cfg.proc.epoch_to_process[0]}] - {epoch_map[ep].timestamp}..."""  # noqa: E501
    )
    iter += 1
    epochdir = cfg.paths.results_dir / epoch_map.get_timestamp_str(ep)
    match_dir = epochdir / "matching"

    # Load existing epcoh
    if cfg.proc.load_existing_results:
        try:
            epoch = Epoch.read_pickle(
                epochdir / f"{epoch_map.get_timestamp(ep)}.pickle"
            )

            # Compute reprojection error
            io.write_reprojection_error_to_file(cfg.residuals_fname, epoches[ep])

            # Save focal length to file
            io.write_cameras_to_file(cfg.camera_estimated_fname, epoches[ep])

            continue
        except:
            logger.error(
                f"Unable to load epoch {epoch_map.get_timestamp(ep)} from pickle file. Creating new epoch..."
            )
            epoch = initialization.initialize_epoch(
                cfg=cfg,
                images=epoch_map.get_images(ep),
                epoch_id=ep,
                epoch_dir=epochdir,
            )

    else:
        epoch = initialization.initialize_epoch(
            cfg=cfg,
            epoch_timestamp=epoch_map.get_timestamp(ep),
            images=epoch_map.get_images(ep),
            epoch_dir=epochdir,
        )

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


    matcher = matching.LightGlueMatcher()
    matcher.match(
        epoch.images[cams[0]].value,
        epoch.images[cams[1]].value,
        quality=matching.Quality.HIGH,
        tile_selection= matching.TileSelection.PRESELECTION,
        grid=[2, 2],
        overlap=200,
        origin=[0, 0],
        do_viz_matches=True,
        do_viz_tiles=True,
        min_matches_per_tile = 3,
        max_keypoints = 8196,    
        save_dir=epoch.epoch_dir / "matching",
        geometric_verification=matching.GeometricVerification.PYDEGENSAC,
        threshold=2,
        confidence=0.9999,
    )
    # # Define matching parameters
    # matching_quality = matching.Quality.HIGH
    # tile_selection = matching.TileSelection.PRESELECTION
    # tiling_grid = [4, 3]
    # tiling_overlap = 200
    # geometric_verification = matching.GeometricVerification.PYDEGENSAC
    # geometric_verification_threshold = 1
    # geometric_verification_confidence = 0.9999
    # Create a new matcher object
    # matcher = matching.SuperGlueMatcher(cfg.matching)
    # matcher.match(
    #     epoch.images[cams[0]].value,
    #     epoch.images[cams[1]].value,
    #     quality=matching_quality,
    #     tile_selection=tile_selection,
    #     grid=tiling_grid,
    #     overlap=tiling_overlap,
    #     do_viz_matches=True,
    #     do_viz_tiles=False,
    #     save_dir=match_dir,
    #     geometric_verification=geometric_verification,
    #     threshold=geometric_verification_threshold,
    #     confidence=geometric_verification_confidence,
    # )
    timer.update("matching")

    # TODO: implement this as a method of Matcher class
    f = {cam: icecore.Features() for cam in cams}
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
    #     logger.info("Performing additional matching on user-specified patches")
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
    #     logger.info("Matching by patches completed.")

    timer.update("matching")

    """ SfM """

    logger.info(f"Reconstructing epoch {ep}...")

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
        compute_colors=True, image=epoch.images[cams[1]].value, cam_id=1
    )
    logger.info("Tie points triangulated.")

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
            ), f"""epoch {ep} - {epoch_map.get_timestamp(ep)}: 
            Different targets found in image {id} - {epoch.images[cams[id]]}"""
        if len(valid_targets) < 1:
            logger.error(
                f"Not enough targets found. Skipping epoch {ep} and moving to next epoch"  # noqa: E501
            )
            continue
        if valid_targets != cfg.georef.targets_to_use:
            logger.warning(f"Not all targets found. Using onlys {valid_targets}")

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
            _ = abs_ori.estimate_transformation_linear(estimate_scale=True)
            points3d = abs_ori.apply_transformation(points3d=points3d)
            for i, cam in enumerate(cams):
                epoch.cameras[cam] = abs_ori.cameras[i]
            logger.info("Absolute orientation completed.")
        except ValueError as err:
            logger.error(
                f"""{err}. Absolute orientation failed. 
                Not enough targets available.
                Skipping epoch {ep} and moving to next epoch"""
            )
            continue

    # save_to_colmap()

    # Create point cloud and save .ply to disk
    # pcd_epc = icecore.PointCloud(points3d=points3d, points_col=triang.colors)
    pts = icecore.Points()
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
            logger.warning(
                f"""Metashape folder {metashape_path} already exists,
                but force_overwrite_projects is set to True.
                Removing all old Metashape files"""
            )
            shutil.rmtree(metashape_path, ignore_errors=True)

        # Export results in Bundler format
        io.ICEcy4D_2_metashape(
            export_dir=epochdir,
            images=epoch.images,
            cameras=epoch.cameras,
            features=epoch.features,
            points=pts,
            targets=epoch.targets,
            targets_to_use=valid_targets,
            targets_enabled=[True for el in valid_targets],
        )

        ms_cfg = MS.build_metashape_cfg(cfg, epoch.timestamp)
        im_list = [epoch.images[cam].path for cam in cams]
        metashape = MS.MetashapeProject(im_list, ms_cfg, timer)
        metashape.run_full_workflow()

        ms_reader = MS.MetashapeReader(
            metashape_dir=epochdir / "metashape",
            num_cams=len(cams),
        )
        ms_reader.read_icepy4d_outputs()
        # for i, cam in enumerate(cams):
        #     focals[cam][ep] = ms_reader.get_focal_lengths()[i]

        # Assign camera extrinsics and intrinsics estimated in Metashape to
        # Camera Object

        img_stems = [epoch.images[cam].stem for cam in cams]
        ms_label = list(ms_reader.extrinsics.keys())
        cam_idx_map = {cam: ms_label.index(stem) for cam, stem in zip(cams, img_stems)}
        cam_label_map = {cam: ms_label[idx] for cam, idx in cam_idx_map.items()}
        for cam_idx, cam in enumerate(cams):
            epoch.cameras[cam].update_K(ms_reader.K[cam_idx])
        for cam, label in cam_label_map.items():
            epoch.cameras[cam].update_extrinsics(ms_reader.extrinsics[label])

        # Triangulate again points and save point cloud to disk
        triang = sfm.Triangulate(
            [epoch.cameras[cams[0]], epoch.cameras[cams[1]]],
            [
                epoch.features[cams[0]].kpts_to_numpy(),
                epoch.features[cams[1]].kpts_to_numpy(),
            ],
        )
        points3d = triang.triangulate_two_views(
            compute_colors=True,
            image=epoch.images[cams[1]].value,
            cam_id=1,
        )

        epoch.points.append_points_from_numpy(
            points3d,
            track_ids=epoch.features[cams[0]].get_track_ids(),
            colors=triang.colors,
        )

        if cfg.proc.save_sparse_cloud:
            epoch.points.to_point_cloud().write_ply(
                cfg.paths.results_dir
                / f"point_clouds/sparse_{epoch_map.get_timestamp(ep)}.ply"
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
        del ms_cfg, metashape, ms_reader
        gc.collect()

        # Save epoch as a pickle object
        epoches[ep].save_pickle(f"{epochdir}/{epoch_map.get_timestamp(ep)}.pickle")

        # Save matches plot
        matches_fig_dir = "res/fig_for_paper/matches_fig"
        # plot_matches_epoch(epoches[ep], ep, matches_fig_dir, show_fig=False)

        # Compute reprojection error and save to file
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

    logger.info("Performing homograpy warping for DIC")

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
            image=epoch.images[cam].value,
            undistort=True,
            out_path=f"res/warped/{epoch.images[cam].name}",
        )

    timer_global.update("Homograpy warping")

timer_global.print("Total time elapsed")

logger.info("Processing completed.")
