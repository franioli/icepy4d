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

import open3d as o3d
import numpy as np
import logging
import laspy

from pathlib import Path
from typing import Union


class PointCloud:
    """
    Class that wraps around an Open3D point cloud object.

    TODO: implement metadata (e.g. from las file) and scalar fields.
    """

    def __init__(
        self,
        points3d: np.ndarray = None,
        pcd_path: str = None,
        points_col: np.ndarray = None,
        # *scalar_fied: np.ndarray = None,
        verbose: bool = False,
    ) -> None:
        """
        __init__ _summary_

        Args:
            points3d (np.ndarray, optional): _description_. Defaults to None.
            pcd_path (str, optional): _description_. Defaults to None.
            points_col (np.ndarray, optional): _description_. Defaults to None.
            verbose (bool, optional): _description_. Defaults to False.

        TODO: implement support for laspy
        TODO: additional wrapper for plotting point clouds

        """
        if isinstance(points3d, np.ndarray):
            self.from_numpy(points3d, points_col)
        elif pcd_path is not None:
            o3d_format = [".ply", ".pcd", ".txt", ".csv"]
            pcd_path = Path(pcd_path)
            if any(pcd_path.suffix in e for e in [".las", ".laz"]):
                self.read_las(pcd_path)
            elif any(pcd_path.suffix in e for e in o3d_format):
                self.pcd = o3d.io.read_point_cloud(str(pcd_path))
            else:
                logging.error(
                    "Invalid file format. It mus be a las (it uses laspy) or one among [.ply, .pcd, .txt, .csv] (it uses open3d)"
                )
        self._verbose = verbose

    def __repr__(self):
        return f"PointCloud with {len(self)} points"

    def __len__(self):
        return len(self.pcd.points)

    def get_pcd(self) -> o3d.geometry.PointCloud:
        """Get Open3d object"""
        return self.pcd

    def get_points(self) -> np.ndarray:
        """Get point coordinates as nx3 numpy array"""
        return np.asarray(self.pcd.points)

    def get_colors(self, as_float: bool = False) -> np.ndarray:
        """Get point colors as nx3 numpy array of integers values (0-255)"""
        if as_float:
            return self.pcd.colors.astype(np.float32)
        else:
            return (np.asarray(self.pcd.colors) * 255.0).astype(int)

    def read_las(self, path: Union[str, Path]):
        """
        read_las Read las point cloud and extract points coordinates.

        Args:
            path (Union[str, Path]): path to the point cloud

        TODO:
            read also metadata, scalar fields, normals etc.
        """
        try:
            las = laspy.read(path)
        except:
            logging.error(f"Unable to read {path.name}.")
            raise ValueError(f"Unable to read {path.name}.")
        self.from_numpy(points3d=las.xyz)

    def from_numpy(
        self,
        points3d: np.ndarray,
        points_col: np.ndarray = None,
        # *scalar_fied: np.ndarray,
    ) -> None:
        """
        Creates a point cloud object from numpy array using Open3D library.

        Args:
            points3d (np.ndarray): A numpy array of shape (n, 3) with float32 dtype containing the 3D points.
            points_col (Optional[np.ndarray]): A numpy array of shape (n, 3) with float32 dtype containing the color of each point. Colors are defined in the range [0, 1] as float numbers. Defaults to None.
            scalar_fied (Tuple[np.ndarray]): Tuple of numpy arrays representing scalar fields. To be implemented. Defaults to empty tuple.

        Returns:
            None

        TODO:
            implement scalar fields.

        """
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points3d)
        if points_col is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(points_col)

    def sor_filter(self, nb_neighbors: int = 10, std_ratio: float = 3.0):
        _, ind = self.pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        self.pcd = self.pcd.select_by_index(ind)
        if self._verbose:
            logging.info("Point cloud filtered by Statistical Oulier Removal")

    def write_ply(self, path: Union[str, Path]) -> True:
        """Write point cloud to disk as .ply format.

        Args:
            path (Union[str, Path]): Path or string of the file where to save the point cloud to disk in .ply format.

        Returns:
            bool. Returns True if the point cloud is successfully saved to disk.

        Raises:
            IOError: If the point cloud could not be saved to disk.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(path), self.pcd)

        return True

    def write_las(self, path: Union[str, Path]) -> bool:
        """Not working yet. Write point cloud to disk as .las format."""

        points = np.asarray(self.pcd.points)
        header = laspy.LasHeader(point_format=3, version="1.4")
        header.offsets = np.min(points, axis=0)
        header.scales = np.array([1, 1, 1])
        with laspy.open(str(path), mode="w", header=header) as writer:
            point_record = laspy.ScaleAwarePointRecord.zeros(
                points.shape[0], header=header
            )
            point_record.x = points[:, 0]
            point_record.y = points[:, 1]
            point_record.z = points[:, 2]

            writer.write_points(point_record)

        return True


if __name__ == "__main__":
    pass

    pcd = PointCloud(pcd_path="res/point_clouds/dense_2022_05_01.ply")

    pcd.write_las("test.las")
