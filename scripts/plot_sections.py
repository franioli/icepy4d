import logging
from pathlib import Path

import cloudComPy as cc
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D, art3d
from tqdm import tqdm

from icepy4d.post_processing.open3d_fun import (
    filter_pcd_by_polyline,
    read_and_merge_point_clouds,
)


def cc_to_o3d(pcd):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.toNpArrayCopy())
    return o3d_pcd


PCD_DIR = "res/monthly_pcd/pcd_loc"
PCD_PATTERN = "*.ply"

pcd_list = sorted(Path(PCD_DIR).glob(PCD_PATTERN))

sec_bin_file = "res/monthly_pcd/sections/aa.bin"
res = cc.importFile(sec_bin_file)
clouds = res[1]
names = [cloud.getName().split()[0] for cloud in clouds]
secs_cc = {cloud.getName().split()[0]: cloud for cloud in clouds}
secs_np = {k: pcd.toNpArrayCopy() for k, pcd in secs_cc.items()}
secs_o3d = [cc_to_o3d(pcd) for pcd in secs_cc.values()]


# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


# Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
for name, cloud in secs_np.items():
    ax.scatter(
        cloud[:, 0], cloud[:, 1], cloud[:, 2], label=name, s=5, alpha=0.7
    )  # Increase point size and alpha value
ax.set_xlabel("X", fontsize=12)  # Increase label fontsize
ax.set_ylabel("Y", fontsize=12)
ax.set_zlabel("Z", fontsize=12)
ax.legend(
    prop={"size": 12}, markerscale=4
)  # Increase legend symbol size and marker scale
ax.view_init(elev=0, azim=-90)  # Set the view on the XZ plane
ax.set_box_aspect([1, 1, 1])  # Set regular spacing for grid
ax.set_proj_type("ortho")  # Use orthographic projection for better visibility
set_axes_equal(ax)  # Set equal axis scale
ax.tick_params(axis="both", which="major", labelsize=10)  # Adjust tick label size
ax.grid(True, linestyle="--", alpha=0.5)  # Add grid with dashed lines
plt.tight_layout()  # Improve spacing between subplots
plt.show()


# Viz with open3d
# o3d.visualization.draw_geometries(secs_o3d)


print("Done")
