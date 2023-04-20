import logging
import multiprocessing
from itertools import compress
from pathlib import Path
from typing import Union, List

import numpy as np
import open3d as o3d
from easydict import EasyDict as edict
from matplotlib import path as mpath

from icepy4d.utils.spatial_funs import ccw_sort_points, point_in_hull
from icepy4d.utils.timer import AverageTimer

# def filter_mesh_by_convex_hull(mesh, pcd):
#     convex_hull = pcd.compute_convex_hull()

#     keep = point_in_hull(np.asarray(mesh.vertices), np.asarray(convex_hull.vertices))
#     idx = list(compress(range(len(keep)), keep))
#     mesh = mesh.select_by_index(idx)

#     return mesh


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries(
        [inlier_cloud, outlier_cloud],
    )


def filter_pcd_by_polyline(
    pcd: o3d.geometry.PointCloud,
    polyline_path: str,
    dir: str = "x",
) -> o3d.geometry.PointCloud:
    """
    Filters a point cloud based on a given polyline in the X-Y plane.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        polyline_path (str): The path to a text file containing the polyline coordinates.
        dir (str, optional): The plane on which to cut the point cloud. Can be "x", "y" or "z". Defaults to "x".

    Returns:
        o3d.geometry.PointCloud: The filtered point cloud.

    Raises:
        ValueError: If the given direction is not "x", "y" or "z".

    NOTE:
        The polyline is defined by a set of points, which are loaded from a text file specified by polyline_path. The polyline is sorted in a counterclockwise order and closed to form a polygon. The point cloud is then filtered based on whether the point lies inside or outside the polygon in the specified plane.
    """
    with open(polyline_path, "r") as f:
        poly = np.loadtxt(f, delimiter=" ")
    if dir == "x":
        poly = poly[:, 1:]
    else:
        raise ValueError("Cutting point cloud implemented only on Y-Z plane")

    poly_sorted = ccw_sort_points(poly)
    poly_sorted = np.concatenate((poly_sorted, poly_sorted[0, :].reshape(1, 2)), axis=0)
    codes = [mpath.Path.LINETO for row in poly_sorted]
    codes[0] = mpath.Path.MOVETO
    codes[-1] = mpath.Path.CLOSEPOLY
    polygon = mpath.Path(poly_sorted, codes)

    if dir == "x":
        points = np.asarray(pcd.points)[:, 1:]
    keep = polygon.contains_points(points)
    idx = list(compress(range(len(keep)), keep))
    pcd_out = pcd.select_by_index(idx)

    return pcd_out


def read_and_merge_point_clouds(pcd_names: List[str]) -> o3d.geometry.PointCloud:
    """Merge multiple point clouds into a single point cloud.

    Args:
        pcd_names (List[str]): A list of file paths of point clouds to merge.

    Returns:
        o3d.geometry.PointCloud: A merged point cloud containing all points from input point clouds.

    Raises:
        FileNotFoundError: If any of the input point cloud files do not exist.

    Notes:
        This function reads multiple point cloud files specified by file paths in `pcd_names`, and merges them into a single point cloud. The merged point cloud contains all points from the input point clouds, with the corresponding colors.
    """
    input_pcds = {}
    for k, path in enumerate(pcd_names):
        if not Path(path).is_file():
            raise FileNotFoundError(f"File not found: {path}")
        input_pcds[k] = o3d.io.read_point_cloud(path)

    tot_points = sum(len(pcd.points) for pcd in input_pcds.values())
    pts_all = np.empty((tot_points, 3))
    col_all = np.empty((tot_points, 3))

    idx = 0
    for pcd in input_pcds.values():
        pts_all[idx : idx + len(pcd.points)] = np.asarray(pcd.points)
        col_all[idx : idx + len(pcd.points)] = np.asarray(pcd.colors)
        idx += len(pcd.points)

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(pts_all)
    merged.colors = o3d.utility.Vector3dVector(col_all)

    return merged


class MeshingPoisson:
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
            out_dir (Union[str, Path]): Path of the output folder
            cfg (dict): configuration dictionary with the following keys:
                    {
                        "do_SOR" (bool): filter point cloud by SOR (default = True),
                        "poisson_depth" (int): Depth parameter for poisson meshing (default = 9),
                        "min_mesh_denisty" (int): minimum density of the mesh triangles for filtering mesh by density (default = 11)
                        "save_mesh" (bool): Save mesh to disk (default = True)
                        "sample_mesh" (bool): Sample mesh with uniform distribution (default = True)
                        "num_sampled_points" (int): number of points to sample (default = 2 * 10**6)
                        "crop_polyline_path" (str): path to the polyline for cropping the sampled point cloud.
                    }
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
        self.logger = logging.getLogger(multiprocessing.current_process().name)
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
        keep = point_in_hull(
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

        poly_sorted = ccw_sort_points(poly)
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
        # NOTE: filter_mesh_by_density() MUST be callded before filter_mesh_by_convex_hull() because of a bug in the density array when filtering the mesh first. Fix it
        self.read_pcd()
        self.poisson_meshing()
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


if __name__ == "__main__":

    PCD_DIR = "res/point_clouds"
    PCD_PATTERN = "dense_2022*.ply"
    OUT_DIR = "res/point_clouds_meshed"

    LOG_LEVEL = logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
        level=LOG_LEVEL,
    )

    pcd_list = sorted(Path(PCD_DIR).glob(PCD_PATTERN))
    n = len(pcd_list)

    polyline_path = "test/poly.poly"
    output_dir = "test"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(pcd_list[0]))
    cropped = filter_pcd_by_polyline(pcd, polyline_path, dir="x")
    o3d.io.write_point_cloud(str(output_dir / f"test.ply"), cropped)

    # Test meshing
    CFG = {
        "do_SOR": False,
        "poisson_depth": 9,
        "min_mesh_denisty": 9,
        "save_mesh": False,
        "sample_mesh": True,
        "num_sampled_points": 4 * 10**6,
        "crop_polyline_path": "data/crop_polyline.poly",
    }
    m = MeshingPoisson(pcd_list[0], out_dir=output_dir, cfg=CFG)
    if not m.run():
        raise RuntimeError(f"Unable to mesh point cloud {pcd_list[0]}")

    print("done.")
