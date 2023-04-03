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

import logging
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

#%%
import numpy as np
from matplotlib import pyplot as plt

# ICEpy4D
import icepy.classes as icepy_classes
import icepy.utils as icepy_utils
import icepy.utils.initialization as initialization
import icepy.visualization as icepy_viz
from icepy.classes.solution import Solution

matplotlib.use("TkAgg")


cfg_file = Path("config/config_2022_exp.yaml")

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

    path = epochdir / f"{epoch_dict[epoch]}.pickle"
    logging.info(f"Loading solution from {path}")
    solution = Solution.read_solution(path, ignore_errors=True)
    if solution is not None:
        cameras[epoch], _, features[epoch], points[epoch] = solution
        # logging.info("Solution imported.")

        # # TMP fct to export to h5
        # from collections import defaultdict
        # from copy import deepcopy

        # import h5py
        # import torch

        # MIN_MATCHES = 20

        # def features_to_h5(features: icepy_classes.FeaturesDictEpoch) -> bool:
        #     key1, key2 = images[cams[0]][epoch], images[cams[1]][epoch]

        #     mkpts0 = features[epoch][cams[0]].kpts_to_numpy()
        #     mkpts1 = features[epoch][cams[1]].kpts_to_numpy()
        #     n_matches = len(mkpts0)

        #     output_dir = Path(epochdir)
        #     db_name = output_dir / f"{epoch_dict[epoch]}.h5"
        #     with h5py.File(db_name, mode="w") as f_match:
        #         group = f_match.require_group(key1)
        #         if n_matches >= MIN_MATCHES:
        #             group.create_dataset(
        #                 key2, data=np.concatenate([mkpts0, mkpts1], axis=1)
        #             )
        #     kpts = defaultdict(list)
        #     match_indexes = defaultdict(dict)
        #     total_kpts = defaultdict(int)

        #     with h5py.File(db_name, mode="r") as f_match:
        #         for k1 in f_match.keys():
        #             group = f_match[k1]
        #             for k2 in group.keys():
        #                 matches = group[k2][...]
        #                 total_kpts[k1]
        #                 kpts[k1].append(matches[:, :2])
        #                 kpts[k2].append(matches[:, 2:])
        #                 current_match = (
        #                     torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
        #                 )
        #                 current_match[:, 0] += total_kpts[k1]
        #                 current_match[:, 1] += total_kpts[k2]
        #                 total_kpts[k1] += len(matches)
        #                 total_kpts[k2] += len(matches)
        #                 match_indexes[k1][k2] = current_match

        #     for k in kpts.keys():
        #         kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
        #     unique_kpts = {}
        #     unique_match_idxs = {}
        #     out_match = defaultdict(dict)
        #     for k in kpts.keys():
        #         uniq_kps, uniq_reverse_idxs = torch.unique(
        #             torch.from_numpy(kpts[k]), dim=0, return_inverse=True
        #         )
        #         unique_match_idxs[k] = uniq_reverse_idxs
        #         unique_kpts[k] = uniq_kps.numpy()
        #     for k1, group in match_indexes.items():
        #         for k2, m in group.items():
        #             m2 = deepcopy(m)
        #             m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
        #             m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
        #             out_match[k1][k2] = m2.numpy()
        #     with h5py.File(output_dir / f"keypoints.h5", mode="w") as f_kp:
        #         for k, kpts1 in unique_kpts.items():
        #             f_kp[k] = kpts1

        #     with h5py.File(output_dir / f"matches.h5", mode="w") as f_match:
        #         for k1, gr in out_match.items():
        #             group = f_match.require_group(k1)
        #             for k2, match in gr.items():
        #                 group[k2] = match

        del solution
        logging.info("Solution loaded.")
        continue
    else:
        logging.error("Unable to import solution.")

#%%

"""Tests"""

# import open3d as o3d
# from scipy.spatial import KDTree

# from icepy.tracking_features_utils import *

# folder_out = Path("test_out")
# folder_out.mkdir(parents=True, exist_ok=True)
# viz = False
# save_figs = False
# min_dt = 2

# fdict = sort_features_by_cam(features, cams[0])
# # bbox = np.array([800, 1500, 5500, 2500])
# vol = np.array(
#     [
#         [0.0, 120.0, 110.0],
#         [0.0, 340.0, 110.0],
#         [-150.0, 120.0, 110.0],
#         [-150.0, 340.0, 110.0],
#         [0.0, 120.0, 140.0],
#         [0.0, 340.0, 140.0],
#         [-150.0, 120.0, 140.0],
#         [-150.0, 340.0, 140.0],
#     ]
# )
# # fts = tracked_features_time_series(
# #     fdict,
# #     min_tracked_epoches=2,
# #     rect=bbox,
# # )
# fts = tracked_points_time_series(points, min_tracked_epoches=min_dt, volume=vol)
# fts_df = tracked_dict_to_df(
#     features,
#     points,
#     epoch_dict,
#     fts,
#     min_dt=min_dt,
#     vx_lims=[0, 0.3],
#     vy_lims=[-0.05, 0.05],
#     vz_lims=[-0.2, 0],
#     save_path=folder_out / "test.csv",
# )
# logging.info("Time series of tracked points and onverted to pandas df")

# Binned stats

# from icepy.binned_stats import *

# # Get points and compute displacements
# pts0 = fts_df[["X_ini", "Y_ini", "Z_ini"]].to_numpy()
# displacement = fts_df[["dX", "dY", "dZ"]].to_numpy()
# magnitude = np.linalg.norm(displacement, axis=1)
# unit_vector = displacement / magnitude[:, np.newaxis]

# # Define grid
# step = 10
# xmin, ymin, xmax, ymax = (
#     vol[:, 0].min(),
#     vol[:, 1].min(),
#     vol[:, 0].max(),
#     vol[:, 1].max(),
# )
# x_nodes = np.arange(xmin, xmax, step)
# y_nodes = np.arange(ymin, ymax, step)

# xnode, ynode, magnitude_binned = compute_binned_stats(
#     pts0[:, 0:2],
#     displacement.reshape(-1, 1),
#     statistic="median",
#     x_nodes=x_nodes,
#     y_nodes=y_nodes,
#     display_results=True,
# )
# _, _, dx = compute_binned_stats(
#     pts0[:, 0:2],
#     displacement[:, 0:1],
#     statistic="median",
#     x_nodes=x_nodes,
#     y_nodes=y_nodes,
# )
# _, _, dy = compute_binned_stats(
#     pts0[:, 0:2],
#     displacement[:, 1:2],
#     statistic="median",
#     x_nodes=x_nodes,
#     y_nodes=y_nodes,
# )
# _, _, dz = compute_binned_stats(
#     pts0[:, 0:2],
#     displacement[:, 2:3],
#     statistic="median",
#     x_nodes=x_nodes,
#     y_nodes=y_nodes,
# )


""" Tracking all features for one week around 26 July 2022"""

import open3d as o3d

from icepy.tracking_features_utils import *
from icepy.utils.rototranslation import Rotrotranslation, belvedere_loc2utm

belv_rotra = Rotrotranslation(belvedere_loc2utm())

min_dt = 1

folder_out = Path("test_out")
folder_out.mkdir(parents=True, exist_ok=True)
fts = tracked_points_time_series(points, min_tracked_epoches=min_dt)
fts_df = tracked_dict_to_df(
    features,
    points,
    epoch_dict,
    fts,
    min_dt=min_dt,
    vx_lims=[0, 0.4],
    vy_lims=[-0.10, 0.10],
    vz_lims=[-0.3, 0],
    save_path=folder_out / "test.csv",
)
logging.info("Time series of tracked points and converted to pandas df")

pts_utm = belv_rotra.apply_transformation(
    fts_df[["X_ini", "Y_ini", "Z_ini"]].to_numpy().T
).T
fts_df["East_ini"] = pts_utm[:, 0]
fts_df["North_ini"] = pts_utm[:, 1]
fts_df["h_ini"] = pts_utm[:, 2]
pts_utm = belv_rotra.apply_transformation(
    fts_df[["X_fin", "Y_fin", "Z_fin"]].to_numpy().T
).T
fts_df["East_fin"] = pts_utm[:, 0]
fts_df["North_fin"] = pts_utm[:, 1]
fts_df["h_fin"] = pts_utm[:, 2]

fts_df["dE"] = fts_df["East_fin"] - fts_df["East_ini"]
fts_df["dN"] = fts_df["North_fin"] - fts_df["North_ini"]
fts_df["dh"] = fts_df["h_fin"] - fts_df["h_ini"]
fts_df["vE"] = fts_df["dE"] / fts_df["dt"].dt.days
fts_df["vN"] = fts_df["dN"] / fts_df["dt"].dt.days
fts_df["vh"] = fts_df["dh"] / fts_df["dt"].dt.days
fts_df["V"] = np.linalg.norm(fts_df[["vX", "vY", "vZ"]].to_numpy(), axis=1).reshape(
    -1, 1
)
fts_df.to_csv(folder_out / "tracked_points_utm.csv")

# Binned stats

from icepy.binned_stats import *

# Get points and compute displacements
pts0 = fts_df[["East_fin", "North_fin", "h_ini"]].to_numpy()
velocity = fts_df[["vE", "vN", "vh"]].to_numpy()
magnitude = fts_df["V"].to_numpy()

# Define grid
step = 5
buffer = 20
xmin, ymin, xmax, ymax = (
    pts0[:, 0].min() - buffer,
    pts0[:, 1].min() - buffer,
    pts0[:, 0].max() + buffer,
    pts0[:, 1].max() + buffer,
)
x_nodes = np.arange(xmin, xmax, step)
y_nodes = np.arange(ymin, ymax, step)

# Compute binned median
nodesE, nodesN, V = compute_binned_stats(
    pts0[:, 0:2],
    magnitude.reshape(-1, 1),
    statistic="median",
    x_nodes=x_nodes,
    y_nodes=y_nodes,
    display_results=True,
)
_, _, vE = compute_binned_stats(
    pts0[:, 0:2],
    velocity[:, 0:1],
    statistic="median",
    x_nodes=x_nodes,
    y_nodes=y_nodes,
)
_, _, vN = compute_binned_stats(
    pts0[:, 0:2],
    velocity[:, 1:2],
    statistic="median",
    x_nodes=x_nodes,
    y_nodes=y_nodes,
)
_, _, vh = compute_binned_stats(
    pts0[:, 0:2],
    velocity[:, 2:3],
    statistic="median",
    x_nodes=x_nodes,
    y_nodes=y_nodes,
)

msk = np.invert(np.isnan(V.flatten()))
binned_df = pd.DataFrame(
    {
        "nodeE": nodesE.flatten()[msk],
        "nodeN": nodesN.flatten()[msk],
        "vE": vE.flatten()[msk],
        "vN": vN.flatten()[msk],
        "vh": vh.flatten()[msk],
        "V": V.flatten()[msk],
    }
)
binned_df.to_csv(folder_out / "tracked_points_binned_utm.csv")


# downsample_stp = 2
# dense = o3d.io.read_point_cloud("test_out/dense.ply")
# xyz = np.asarray(dense.voxel_down_sample(downsample_stp).points)
fig, ax = plt.subplots()
# ax.plot(xyz[:, 0], xyz[:, 1], ".", color=[0.7, 0.7, 0.7], markersize=0.5, alpha=0.8)
ax.axis("equal")
quiver = ax.quiver(
    xnode,
    ynode,
    dx,
    dy,
    magnitude_binned,
    scale=5,
)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_aspect("equal", "box")
cbar = plt.colorbar(quiver)
cbar.set_label("epoch of detection")
fig.tight_layout()
plt.show()


#######

# Binned statistics from single weeks


def extract_df_fields_from_epoch_range(
    df: pd.DataFrame, ep_st: int, ep_fin: int, fields: List[str]
) -> np.ndarray:
    return fts_df[(fts_df["ep_ini"] >= ep_st) & (fts_df["ep_ini"] < ep_fin)][
        fields
    ].to_numpy()


# Define grid
step = 5
xmin, ymin, xmax, ymax = (
    vol[:, 0].min(),
    vol[:, 1].min(),
    vol[:, 0].max(),
    vol[:, 1].max(),
)
x_nodes = np.arange(xmin, xmax, step)
y_nodes = np.arange(ymin, ymax, step)

dt = 5
magnitude_binned = {}
for ep in cfg.proc.epoch_to_process[:-dt]:
    pts = extract_df_fields_from_epoch_range(
        fts_df, ep, ep + dt, fields=["X_ini", "Y_ini", "Z_ini"]
    )
    displacement = extract_df_fields_from_epoch_range(
        fts_df, ep, ep + dt, fields=["dX", "dY", "dZ"]
    )
    vel = extract_df_fields_from_epoch_range(
        fts_df, ep, ep + dt, fields=["vX", "vY", "vZ", "V"]
    )
    xnode, ynode, magnitude_binned[ep] = compute_binned_stats(
        pts[:, 0:2],
        vel[:, 3:4],
        statistic="median",
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        # display_results=True,
    )

xx_nodes, yy_nodes = np.meshgrid(x_nodes, y_nodes)
plot_binned_stats(x_nodes=xx_nodes, y_nodes=yy_nodes, values=magnitude_binned[190])


def get_point_TS(
    xx_nodes: np.ndarray, yy_nodes: np.ndarray, values: dict, coord: Tuple
):
    msk = np.logical_and(xx_nodes == coord[0], yy_nodes == coord[1])

    val = {}
    for id, array in values.items():
        val[id] = array[msk]

    return val


coord = (-40.0, 230)
valTS = get_point_TS(
    xx_nodes=xx_nodes, yy_nodes=yy_nodes, values=magnitude_binned, coord=coord
)


def plot_TS(val_TS):
    (
        fig,
        ax,
    ) = plt.subplots()
    ax.plot(list(valTS.keys()), list(valTS.values()))


########

if viz:
    cam = cams[0]
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


""" Various"""


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
