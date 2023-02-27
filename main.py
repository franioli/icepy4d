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

# ICEpy4D
import icepy.classes as icepy_classes
import icepy.sfm as sfm
import icepy.metashape.metashape as MS
import icepy.utils.initialization as initialization
import icepy.utils as icepy_utils
import icepy.visualization as icepy_viz
from icepy.classes.solution import Solution

from icepy.matching.match_by_preselection import match_by_preselection
from icepy.matching.tracking_base import tracking_base
from icepy.matching.matching_base import MatchingAndTracking
from icepy.matching.utils import load_matches_from_disk

from icepy.utils.utils import homography_warping
from icepy.io.export2bundler import write_bundler_out


if __name__ == "__main__":

    initialization.print_welcome_msg()

    cfg_file, log_cfg = initialization.parse_command_line()
    # cfg_file = Path("config/config_test.yaml")

    """ Inizialize Variables """
    # Setup logger
    icepy_utils.setup_logger(
        log_cfg["log_folder"],
        log_cfg["log_name"],
        log_cfg["log_file_level"],
        log_cfg["log_console_level"],
    )

    # Parse configuration file
    logging.info(f"Configuration file: {cfg_file.stem}")
    cfg = initialization.parse_yaml_cfg(cfg_file)

    timer_global = icepy_utils.AverageTimer()

    init = initialization.Inizialization(cfg)
    init.inizialize_icepy()
    cams = init.cams
    images = init.images
    epoch_dict = init.epoch_dict
    cameras = init.cameras
    features = init.features
    targets = init.targets
    points = init.points
    focals = init.focals_dict

    """ Big Loop over epoches """

    logging.info("------------------------------------------------------")
    logging.info("Processing started:")
    timer = icepy_utils.AverageTimer()
    iter = 0  # necessary only for printing the number of processed iteration
    for epoch in cfg.proc.epoch_to_process:

        logging.info("------------------------------------------------------")
        logging.info(
            f"Processing epoch {epoch} [{iter}/{cfg.proc.epoch_to_process[-1]-cfg.proc.epoch_to_process[0]}] - {epoch_dict[epoch]}..."
        )
        iter += 1

        epochdir = Path(cfg.paths.results_dir) / epoch_dict[epoch]
        match_dir = epochdir / "matching"

        # Perform matching and tracking
        do_preselection = False
        if cfg.proc.do_matching:
            if do_preselection:
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
        else:
            try:
                features[epoch] = load_matches_from_disk(match_dir)
            except FileNotFoundError as err:
                logging.exception(err)
                logging.warning("Performing new matching and tracking...")
                if do_preselection:
                    features[epoch] = match_by_preselection(
                        images,
                        features[epoch],
                        cams,
                        epoch,
                        cfg.matching,
                        match_dir,
                        n_tiles=6,
                        n_dist=1.5,
                        viz_results=True,
                    )
                else:
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
                targets[epoch].get_image_coor_by_label(
                    cfg.georef.targets_to_use, cam_id=0
                )[0],
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
        # pcd_epc = icepy_classes.PointCloud(points3d=points3d, points_col=triang.colors)
        pts = icepy_classes.Points()
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

            # pcd_epc = icepy_classes.PointCloud(
            #     points3d=points3d, points_col=triang.colors
            # )

            points[epoch].append_points_from_numpy(
                points3d,
                track_ids=features[epoch][cams[0]].get_track_ids(),
                colors=triang.colors,
            )

            if cfg.proc.save_sparse_cloud:
                points[epoch].to_point_cloud().write_ply(
                    cfg.paths.results_dir
                    / f"point_clouds/sparse_{epoch_dict[epoch]}.ply"
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
                homography_warping(
                    cameras[ep_ini][cam], cameras[epoch][cam], image, out_path, timer
                )

            # Save solution as a pickle object
            solution = Solution(cameras[epoch], images, features[epoch], points[epoch])
            solution.save_solutions(f"{epochdir}/{epoch_dict[epoch]}.pickle")
            del solution

        timer.print(f"Epoch {epoch} completed")

    timer_global.update("SfM")

    # camera_model = "OPENCV"
    # export_solution_to_colmap(
    #     export_dir=epochdir,
    #     im_dict=im_dict,
    #     cameras=cameras[epoch],
    #     features=features[epoch],
    #     points=pts,
    #     camera_model=camera_model,
    # )

    #%%

    """Tests"""

    import open3d as o3d
    from scipy.spatial import KDTree

    from icepy.tracking_features_utils import *

    folder_out = Path("test_out")
    folder_out.mkdir(parents=True, exist_ok=True)
    viz = False
    save_figs = False
    min_dt = 2

    fdict = sort_features_by_cam(features, cams[0])
    # bbox = np.array([800, 1500, 5500, 2500])
    vol = np.array(
        [
            [0.0, 120.0, 110.0],
            [0.0, 340.0, 110.0],
            [-80.0, 120.0, 110.0],
            [-80.0, 340.0, 110.0],
            [0.0, 120.0, 140.0],
            [0.0, 340.0, 140.0],
            [-80.0, 120.0, 140.0],
            [-80.0, 340.0, 140.0],
        ]
    )
    # fts = tracked_features_time_series(
    #     fdict,
    #     min_tracked_epoches=2,
    #     rect=bbox,
    # )
    fts = tracked_points_time_series(points, min_tracked_epoches=min_dt, volume=vol)
    fts_df = tracked_dict_to_df(
        features,
        points,
        epoch_dict,
        fts,
        min_dt=min_dt,
        vx_lims=[0, 0.3],
        vy_lims=[-0.05, 0.05],
        vz_lims=[-0.2, 0],
        save_path=folder_out / "test.csv",
    )
    logging.info("Time series of tracked points and onverted to pandas df")

    # delete = [k for k, v in fts.items() if 180 in v]
    # for key in delete:
    #     del fts[key]

    # if save_figs:
    #     def save_tracked_task():
    #         pass
    #     for fid in fts.keys():
    #         for ep in fts[fid]:
    #             fout = folder_out / f"fid_{fid}_ep_{ep}.jpg"
    #             icepy_viz.plot_feature(
    #                 images[cam].read_image(ep).value,
    #                 fdict[ep][fid],
    #                 save_path=fout,
    #                 hide_fig=True,
    #                 zoom_to_feature=True,
    #                 s=10,
    #                 marker="x",
    #                 c="r",
    #                 edgecolors=None,
    #                 window_size=50,
    #             )
    # plt.close("all")
    if viz:
        fid = 2155
        eps = [181, 182]
        fig, axes = plt.subplots(1, len(eps))
        for ax, ep in zip(axes, eps):
            icepy_viz.plot_feature(
                images[cam].read_image(ep).value,
                features[ep][cam][fid],
                ax=ax,
                zoom_to_feature=True,
                s=10,
                marker="x",
                c="r",
                edgecolors=None,
                window_size=300,
            )

    # plot all the features plot
    # f_tracked: icepy_classes.FeaturesDict = {
    #     cam: icepy_classes.Features() for cam in cams
    # }
    # for fid in fts.keys():
    #     for ep in fts[fid]:
    #         f_tracked[cam].append_feature(features[ep][cam][fid])

    # fig, axes = plt.subplots(1, 2)
    # for ax, cam in zip(axes, cams):
    #     ax = icepy_viz.plot_features(
    #         images[cam].read_image(ep).value,
    #         f_tracked[cam],
    #         ax=ax,
    #         s=10,
    #         marker="x",
    #         c="r",
    #         edgecolors=None,
    #     )

    # Quiver plot
    fig, ax = plt.subplots()
    dense = o3d.io.read_point_cloud("test_out/dense.ply")
    xy = np.asarray(dense.points)[:, 0:2]
    ax.plot(xy[:, 0], xy[:, 1], ".", color=[0.7, 0.7, 0.7], markersize=0.5, alpha=0.8)
    # ax.plot(xy[:, 0], xy[:, 1], "."

    # xy = points[ep].to_numpy()[:, 0:2]
    # ax.plot(xy[:, 0], xy[:, 1], ".", color=[0.7, 0.7, 0.7], markersize=1, alpha=0.8)
    quiver = ax.quiver(
        fts_df["X_ini"],
        fts_df["Y_ini"],
        fts_df["vX"],
        fts_df["vY"],
        fts_df["ep_ini"],
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", "box")
    cbar = plt.colorbar(quiver)
    cbar.set_label("epoch of detection")
    fig.tight_layout()
    plt.show()

    # Week 0:
    ep_st, ep_fin = 181, 184
    eps = {ep: epoch_dict[ep] for ep in range(ep_st, ep_fin)}

    ep = list(eps.keys())[1]
    date = eps[ep]
    pts = fts_df[fts_df["ep_ini"] == ep][["X_ini", "Y_ini", "Z_ini"]].to_numpy()

    # dense = o3d.io.read_point_cloud("test_out/dense.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([points[ep].to_point_cloud().pcd, pcd])

    pts = fts_df[["X_ini", "Y_ini", "Z_ini"]].to_numpy()
    # rng = np.random.default_rng()
    # pts = rng.uniform(vol[:, :2].min(), vol[:, :2].max(), (5, 3))
    # xmin, ymin, xmax, ymax = (
    #     vol[:, 0].min(),
    #     vol[:, 1].min(),
    #     vol[:, 0].max(),
    #     vol[:, 1].max(),
    # )
    # res = 20
    # X, Y = np.meshgrid(np.arange(xmin, xmax, res), np.arange(ymin, ymax, res))

    # from itertools import compress, product
    # nodes = zip(X.flatten(), Y.flatten())
    # lim = [
    #     (node[0] - res / 2, node[1] - res / 2, node[0] + res / 2, node[1] + res / 2)
    #     for node in nodes
    # ]
    # pts_in_rect = lambda input: point_in_rect(input[0], input[1])
    # pts_lims = product(pts, lim)
    # out = list(compress(nodes, map(pts_in_rect, pts_lims)))

    from scipy.stats import binned_statistic_2d

    step = 10
    xmin, ymin, xmax, ymax = (
        vol[:, 0].min(),
        vol[:, 1].min(),
        vol[:, 0].max(),
        vol[:, 1].max(),
    )
    pts = fts_df[["X_ini", "Y_ini", "Z_ini"]].to_numpy()
    x = np.arange(xmin, xmax, step)
    y = np.arange(ymin, ymax, step)
    ret = binned_statistic_2d(
        pts[:, 0].flatten(),
        pts[:, 1].flatten(),
        None,
        "count",
        bins=[x, y],
        expand_binnumbers=True,
    )
    binned_points = list(zip(ret.binnumber[0] - 1, ret.binnumber[1] - 1))

    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, vol[4, 2])
    xyz_grid = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))

    dense = o3d.io.read_point_cloud("test_out/dense.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_grid)
    o3d.visualization.draw_geometries([pcd, dense])

    vx, vy, vz = [], [], []
    for bin in binned_points:
        vx = []

    # for key, group in groupby(L, key_func):
    #     print(f"{key}: {list(group)}")

    # import plotly.graph_objects as go
    # import plotly.figure_factory as ff

    # # Create quiver figure
    # fig = ff.create_quiver(
    #     fts_df["X_ini"],
    #     fts_df["Y_ini"],
    #     fts_df["vX"],
    #     fts_df["vY"],
    #     scale=20,
    #     arrow_scale=0.8,
    #     name="tracked points",
    #     line_width=2,
    # )
    # fig.add_trace(
    #     go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="pcd", marker_size=2, name="points")
    # )
    # fig.update_yaxes(
    #     scaleratio=1,
    # )
    # fig.show()

    """End tests"""

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
        # point_clouds = [
        #     points[epoch].to_point_cloud() for epoch in cfg.proc.epoch_to_process
        # ]
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
                    points[epoch].to_numpy(),
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
