import time
import logging
import numpy as np
import open3d as o3d

from easydict import EasyDict as edict
from pathlib import Path
from typing import List, Tuple, Union
from scipy.spatial import Delaunay
from itertools import compress
from matplotlib import path as mpath
from itertools import repeat
import itertools

import multiprocessing

from multiprocessing import Pool, current_process

from os import getpid

from lib.utils.timer import AverageTimer

MP = False

LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)


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


def ccw_sort(p: np.ndarray) -> np.ndarray:
    """
    ccw_sort Sort 2D points in array in counter-clockwise direction around a baricentric mid point

    Args:
        p (np.ndarray): nx2 numpy array containing 2D points

    Returns:
        np.ndarray: nx2 array with sorted points
    """
    mean = np.mean(p, axis=0)
    d = p - mean
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def meshing_task(pcd_path: Path, out_dir: Path, cfg: dict) -> bool:
    """
    meshing_task Task function for running meshing and sampling with a parallel pool.

    Args:
        pcd_path (Path): path to the current point cloud (is passed by map)
        out_dir (Path): output directory
        cfg (dict): configuration dictionary (see Meshing class)
    """
    meshing = Meshing(pcd_path, out_dir=out_dir, cfg=cfg)
    return meshing.run()


class Meshing:
    def __init__(
        self,
        pcd_path: Union[str, Path],
        out_dir: Union[str, Path],
        cfg: Union[dict, edict],
    ) -> None:
        """
        __init__ Initialize class for meshing point cloud and sampling mesh

        Args:
            pcd_path (Union[str, Path]): Path to the point cloud
            ....

            log_level (str): Log level. Choose between ["debug", "info", "warning", "error"] (default: "info")
        """

        pcd_path = Path(pcd_path)
        assert pcd_path.exists(), "Input point cloud does not exists."
        self.pcd_path = pcd_path

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Options
        default_cfg = {
            "do_SOR": True,
            "poisson_depth": 9,
            "min_mesh_denisty": 11,
            "save_mesh": True,
            "sample_mesh": True,
            "num_sampled_points": 2 * 10**6,
        }
        cfg = {**default_cfg, **{k: v for k, v in cfg.items()}}
        self.cfg = edict(cfg)

        # Logging and timing
        self.logger = logging.getLogger(current_process().name)
        self.timer = AverageTimer(logger=self.logger)

    def read_pcd(self) -> None:
        self.pcd = o3d.io.read_point_cloud(str(self.pcd_path))
        self.logger.info(f"Point cloud {self.pcd_path.stem} imported.")
        self.timer.update("Read pcd")

        if self.cfg.do_SOR:
            self.SOR()
        self.convex_hull, _ = self.pcd.compute_convex_hull()

        if not self.pcd.has_normals():
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30)
            )
            self.logger.warning(f"Point cloud did not have normals. Normals computed.")
            self.timer.update("Normals estimation")

    def SOR(self) -> None:
        self.pcd, ind = self.pcd.remove_statistical_outlier(
            nb_neighbors=50, std_ratio=1.5
        )
        self.logger.info(f"Point cloud filtered with SOR.")
        self.timer.update("SOR")

    def poisson_meshing(self) -> None:
        (
            self.mesh,
            self.density,
        ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.pcd, depth=self.cfg.poisson_depth, width=0, scale=1.1, linear_fit=False
        )
        self.mesh.orient_triangles()
        self.density = np.asarray(self.density)
        # bbox = self.pcd.get_axis_aligned_bounding_box()
        # self.mesh = mesh.crop(bbox)
        self.logger.info(f"Point cloud meshed by Poisson mesh reconstruction.")
        self.timer.update("Poisson meshing")

    def filter_mesh_by_density(self) -> None:
        vertexes_to_remove = list(self.density < self.cfg.min_mesh_denisty)
        self.mesh.remove_vertices_by_mask(vertexes_to_remove)
        self.density = self.density[np.invert(vertexes_to_remove)]
        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_duplicated_triangles()
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_non_manifold_edges()
        self.logger.info(
            f"Poisson mesh filtered by density. Removed vertex with density < {self.cfg.min_mesh_denisty}"
        )
        self.timer.update("Filter by density")

    def filter_mesh_by_convex_hull(self):
        keep = in_hull(
            np.asarray(self.mesh.vertices), np.asarray(self.convex_hull.vertices)
        )
        idx = list(compress(range(len(keep)), keep))
        self.mesh = self.mesh.select_by_index(idx)
        # self.density = self.density[np.array(keep).astype(bool)]
        self.logger.info(
            f"Poisson mesh filtered by original pcd convex hull: removed points {len(idx)}/{len(keep)}"
        )
        self.timer.update("Filter by location")

    def sample_mesh(self) -> None:
        self.sampled_pcd = self.mesh.sample_points_uniformly(
            number_of_points=self.cfg.num_sampled_points
        )

    def filter_pcd_by_polygon(
        self,
        pcd: o3d.geometry.PointCloud,
        dir: str = "x",
        polygon: np.ndarray = None,
    ) -> o3d.geometry.PointCloud:

        with open(self.cfg.crop_polyline_path, "r") as f:
            poly = np.loadtxt(f, delimiter=" ")
        if dir == "x":
            poly = poly[:, 1:]
        else:
            raise ValueError("Cutting point cloud implemented only on Y-Z plane")

        poly_sorted = ccw_sort(poly)
        poly_sorted = np.concatenate(
            (poly_sorted, poly_sorted[0, :].reshape(1, 2)), axis=0
        )
        codes = [mpath.Path.LINETO for row in poly_sorted]
        codes[0] = mpath.Path.MOVETO
        codes[-1] = mpath.Path.CLOSEPOLY
        polygon = mpath.Path(poly_sorted, codes)

        if dir == "x":
            points = np.asarray(pcd.points)[:, 1:]
        keep = polygon.contains_points(points)
        idx = list(compress(range(len(keep)), keep))
        pcd_out = pcd.select_by_index(idx)

        self.logger.info("Point cloud cropped by polyline")
        self.timer.update("crop by polyline")
        return pcd_out

    def write_outputs(self) -> bool:
        idx = self.pcd_path.stem.find("202")
        base_name = self.pcd_path.stem[idx:]
        if self.cfg.save_mesh:
            o3d.io.write_triangle_mesh(
                str(self.out_dir / f"mesh_{base_name}.ply"), self.mesh
            )
        if self.cfg.sample_mesh:
            o3d.io.write_point_cloud(
                str(self.out_dir / f"sampled_{base_name}.ply"), self.sampled_pcd
            )
        self.timer.update("save outputs")

        return True

    def run(self) -> bool:
        self.read_pcd()
        self.poisson_meshing()
        # # TODO: filter_mesh_by_density() MUST be callded before filter_mesh_by_convex_hull() because of a bug in the density array when filtering the mesh first. Fix it
        self.filter_mesh_by_density()
        self.filter_mesh_by_convex_hull()
        if self.cfg.sample_mesh:
            self.sample_mesh()
        if self.cfg.crop_polyline_path:
            self.sampled_pcd = self.filter_pcd_by_polygon(self.sampled_pcd)
        if self.write_outputs():
            self.logger.info("Outputs saved successfully.")
        else:
            self.logger.error("Unable to write outputs to disk.")
        self.timer.print()

        return True


def start_process():
    logging.info(f"Starting {current_process().name} with process {getpid()}")


if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")

    NUM_WORKERS = 8

    PCD_DIR = "volumes/data/pcd"  #
    PCD_PATTERN = "*.ply"
    OUT_DIR = "volumes/res/meshed"

    cfg = {
        "do_SOR": False,
        "poisson_depth": 9,
        "min_mesh_denisty": 9,
        "save_mesh": False,
        "sample_mesh": True,
        "num_sampled_points": 2 * 10**6,
        "crop_polyline_path": "volumes/data/crop_polyline.poly",
    }

    pcd_list = sorted(Path(PCD_DIR).glob(PCD_PATTERN))
    n = len(pcd_list)

    # meshing = Meshing(pcd_list[0], out_dir=OUT_DIR, cfg=cfg)
    # meshing.run()

    if MP:
        p = Pool(
            processes=NUM_WORKERS,
            initializer=start_process,
        )
        result = p.starmap(
            meshing_task,
            zip(pcd_list, list(repeat(OUT_DIR, n)), list(repeat(cfg, n))),
        )
        p.close()
        p.join()

    else:
        for pcd in pcd_list:
            meshing_task(pcd, OUT_DIR, cfg)