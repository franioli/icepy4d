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

from pathlib import Path
from typing import Union


class PointCloud:
    """
    Class that wraps around an Open3D point cloud object.
    """

    def __init__(
        self,
        points3d: np.ndarray = None,
        pcd_path: str = None,
        points_col: np.ndarray = None,
        # *scalar_fied: np.ndarray = None,
        verbose: bool = False,
    ) -> None:

        if points3d is not None:
            self.pcd = self.create_point_cloud(points3d, points_col)
        elif pcd_path is not None:
            self.pcd = o3d.io.read_point_cloud(pcd_path)
        self._verbose = verbose

    # Getters
    def get_pcd(self) -> o3d.geometry.PointCloud:
        """Get Open3d object"""
        return self.pcd

    def get_points(self) -> np.ndarray:
        """Get point coordinates as nx3 numpy array"""
        return np.asarray(self.pcd.points)

    def get_colors(self) -> np.ndarray:
        """Get point colors as nx3 numpy array of integers values (0-255)"""
        return (np.asarray(self.pcd.colors) * 255.0).astype(int)

    def __len__(self):
        return len(self.pcd.points)

    # Methods
    def create_point_cloud(
        self,
        points3d: np.ndarray,
        points_col=None,
        *scalar_fied: np.ndarray,
    ) -> o3d.geometry.PointCloud:
        """
        Creates a point cloud object using Open3D library.

        Args:
            points3d (np.ndarray): A numpy array of shape (n, 3) with float32 dtype containing the 3D points.
            points_col (Optional[np.ndarray]): A numpy array of shape (n, 3) with float32 dtype containing the color of each point. Colors are defined in the range [0, 1] as float numbers. Defaults to None.
            scalar_fied (Tuple[np.ndarray]): Tuple of numpy arrays representing scalar fields. To be implemented. Defaults to empty tuple.

        Returns:
            o3d.geometry.PointCloud: An Open3D point cloud object.

        TODO:
            implement scalar fields.

        """

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points3d)
        if points_col is not None:
            pcd.colors = o3d.utility.Vector3dVector(points_col)

        return pcd

    def sor_filter(self, nb_neighbors: int = 10, std_ratio: float = 3.0):

        _, ind = self.pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        pcd_epc = pcd_epc.select_by_index(ind)
        if self._verbose:
            logging.info("Point cloud filtered by Statistical Oulier Removal")

    def write_ply(self, path: Union[str, Path]) -> None:
        """Write point cloud to disk as .ply format.

        Args:
            path (Union[str, Path]): Path or string of the file where to save the point cloud to disk in .ply format.

        Returns:
            None. The point cloud is saved to disk.

        Raises:
            IOError: If the point cloud could not be saved to disk.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(path), self.pcd)
