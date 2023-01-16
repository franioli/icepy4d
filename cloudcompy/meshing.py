#%%
import gc
import time
import logging

import numpy as np
import pandas as pd
import open3d as o3d

from pathlib import Path
from typing import List, Tuple, Union
from multiprocessing import Pool, current_process
from scipy.spatial import Delaunay
from itertools import compress

from cloudcompy import DOD

NUM_WORKERS = 8

PCD_DIR = "cloudcompy/data"
PCD_PATTERN = "dense*.ply"
TSTEP = 3
VERBOSE = True
GRID_STEP = 0.2
DOD_DIR = "x"
OUT_DIR = "cloudcompy/res"
FOUT = "DOD_res_x_20cm.csv"  # "cloudcompy/DOD_res_x_20cm.csv"  #

POISSON_DEPTH = 9
NUM_POINTS = 5 * 10**6
MAX_DIST = 0.5


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries(
        [inlier_cloud, outlier_cloud],
    )


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`
    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def meshing_task(
    pcd_path: Union[str, Path],
):
    pcd_path = Path(pcd_path)
    assert pcd_path.exists(), "input point cloud does not exists."

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    process = current_process()
    logger.info(f"Child {process.name} - Processing pcd {pcd_path}.")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    logger.info("Point cloud imported.")

    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
    convex_hull, _ = pcd_clean.compute_convex_hull()
    logger.info(
        f"Point cloud filtered with SOR. {len(pcd.points) - len(ind)} points removed."
    )

    if not pcd_clean.has_normals():
        logging.warning("Point cloud does not have normals. Computing normals...")
        pcd_clean.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30)
        )

    logger.info("Computing poisson mesh reconstruction...")
    mesh_poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_clean, depth=POISSON_DEPTH, width=0, scale=1.1, linear_fit=False
    )
    mesh_poisson.orient_triangles()
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_poisson = mesh_poisson.crop(bbox)
    # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(convex_hull)
    # hull_ls.paint_uniform_color((1, 0, 0))
    # o3d.visualization.draw_geometries([mesh_poisson, hull_ls])

    logger.info("Mesh created.")

    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 3 * avg_dist
    # logger.info(f"Average point distance: {avg_dist:.3f}")
    # mesh_poisson = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd, o3d.utility.DoubleVector([radius, radius * 2])
    # )
    # mesh_poisson.orient_triangles()
    # bbox = pcd.get_axis_aligned_bounding_box()

    keep = in_hull(np.asarray(mesh_poisson.vertices), np.asarray(convex_hull.vertices))
    idx = list(compress(range(len(keep)), keep))
    logger.info(
        f"Poisson mesh filtering by convex hull: removed points{len(idx)}/{len(keep)}"
    )
    mesh_filtered = mesh_poisson.select_by_index(idx)
    dummy = o3d.geometry.PointCloud()
    dummy.points = o3d.utility.Vector3dVector(np.asarray(mesh_filtered.vertices))
    distances = np.asarray(dummy.compute_nearest_neighbor_distance())
    keep = distances < MAX_DIST
    idx = list(compress(range(len(keep)), keep))
    logger.info(
        f"Poisson mesh filtering by vertexes max distance: removed points{len(idx)}/{len(keep)}"
    )
    mesh_filtered = mesh_filtered.select_by_index(idx)
    mesh_filtered.remove_degenerate_triangles()
    mesh_filtered.remove_duplicated_triangles()
    mesh_filtered.remove_duplicated_vertices()
    mesh_filtered.remove_non_manifold_edges()
    logger.info("Mesh filtered by original convex hull and vertex distance")

    pcd_sampled_unif = mesh_filtered.sample_points_uniformly(
        number_of_points=NUM_POINTS
    )
    logger.info("Mesh sampled uniformly")

    name = pcd_path.stem.replace("dense", "mesh")
    o3d.io.write_triangle_mesh(str(out_dir / f"{name}.ply"), mesh_poisson)
    name = pcd_path.stem.replace("dense", "sampled")
    o3d.io.write_point_cloud(str(out_dir / f"{name}.ply"), pcd_sampled_unif)
    logger.info(f"Mesh and sampled point cloud exported to {OUT_DIR}")

    # o3d.visualization.draw_geometries([pcd_sampled_unif])

    logger.info(f"Child {process.name} completed.")


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    pcd_dir = Path(PCD_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert pcd_dir.is_dir(), f"Directory '{pcd_dir}' does not exists."
    assert any(pcd_dir.iterdir()), f"Directory '{pcd_dir}' is empty."

    fout = out_dir / FOUT
    if fout.exists():
        logger.warning(f"Output file {fout} already exists. Removing it.")
        fout.unlink()

    pcd_list = sorted(pcd_dir.glob(PCD_PATTERN))

    # meshing_task(pcd_list[0])

    logger.info("DOD computation started:")
    t0 = time.time()
    pool = Pool(processes=NUM_WORKERS)
    pool.map(meshing_task, pcd_list)
    pool.close()
    t1 = time.time()
    logger.info(f"DOD computation completed. Elapsed timMaine: {t1-t0:.2f} sec")
