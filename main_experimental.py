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

import matplotlib

matplotlib.use("TkAgg")


cfg_file = Path("config/config_block_3_4.yaml")

""" Inizialize Variables """
# Setup logger
icepy_utils.setup_logger(
    log_folder="logs",
    log_base_name="icepy_experimental",
    console_log_level="info",
    logfile_level="info",
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

    # Load Existing solution

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

# Quiver 2D plot

# stp = 1
# dense = o3d.io.read_point_cloud("test_out/dense.ply")
# xyz = np.asarray(dense.voxel_down_sample(stp).points)
ep = 182
xyz = points[ep].to_numpy()

if viz:
    fig, ax = plt.subplots()
    ax.plot(xyz[:, 0], xyz[:, 1], ".", color=[0.7, 0.7, 0.7], markersize=0.5, alpha=0.8)
    ax.axis("equal")
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

    # Quiver plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        ".",
        color=[0.7, 0.7, 0.7],
        markersize=2,
        alpha=0.8,
    )
    quiver = ax.quiver(
        fts_df["X_ini"],
        fts_df["Y_ini"],
        fts_df["Z_ini"],
        fts_df["vX"],
        fts_df["vY"],
        fts_df["vZ"],
        length=100,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim([190, -30])
    ax.set_ylim([680, 850])
    ax.set_zlim([-120, 20])
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    plt.show()

# Week 0:
ep_st, ep_fin = 181, 184
eps = {ep: epoch_dict[ep] for ep in range(ep_st, ep_fin)}

ep = list(eps.keys())[1]
date = eps[ep]
pts = fts_df[fts_df["ep_ini"] == ep][["X_ini", "Y_ini", "Z_ini"]].to_numpy()

# dense = o3d.io.read_point_cloud("test_out/dense.ply")
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pts)
# o3d.visualization.draw_geometries([points[ep].to_point_cloud().pcd, pcd])

pts = fts_df[(fts_df["ep_ini"] >= ep_st) & (fts_df["ep_ini"] < ep_fin)][
    ["X_ini", "Y_ini", "Z_ini"]
].to_numpy()
vel = fts_df[(fts_df["ep_ini"] >= ep_st) & (fts_df["ep_ini"] < ep_fin)][
    ["vX", "vY", "vZ"]
].to_numpy()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    ".",
    color=[0.7, 0.7, 0.7],
    markersize=2,
    alpha=0.8,
)
quiver = ax.quiver(
    pts[:, 0],
    pts[:, 1],
    pts[:, 2],
    vel[:, 0],
    vel[:, 1],
    vel[:, 2],
    length=100,
)
ax.set_xlim([190, -30])
ax.set_ylim([680, 850])
ax.set_zlim([-120, 20])
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
# ax.axis("equal")
ax.set_aspect("equal", "box")
fig.tight_layout()
plt.show()

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
pts0 = fts_df[["X_ini", "Y_ini", "Z_ini"]].to_numpy()
pts1 = fts_df[["X_fin", "Y_fin", "Z_fin"]].to_numpy()

x = np.arange(xmin, xmax, step)
y = np.arange(ymin, ymax, step)
ret = binned_statistic_2d(
    pts0[:, 0].flatten(),
    pts0[:, 1].flatten(),
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


displacement = pts1 - pts0
magnitude = np.linalg.norm(displacement, axis=1)
unit_vector = displacement / magnitude[:, np.newaxis]

vx, vy, vz = [], [], []
for bin in binned_points:
    vx = []

pcd = icepy_classes.PointCloud(pcd_path="test_out/dense_2022.ply")
icepy_viz.display_point_cloud([pcd], list(cameras[ep].values()), plot_scale=20)

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
    fig.savefig(cfg.paths.results_dir / f"focal_lenghts_{cfg_file.stem}.png", dpi=100)

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
