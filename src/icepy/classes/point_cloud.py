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
        """Function to create a point cloud object by using Open3D library.
        ---------
        Parameters:
        - points3d (nx3, float32): array of points 3D.
        - points_col (nx3, float32): array of color of each point.
                    Colors are defined in [0,1] range as float numbers.
        - Scalar fields: to be implemented. #@TODO: implement scalar fields.
        Return: Open3D point cloud object
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
        """Write point cloud to disk as .ply

        Parameters
        ----------
        pcd : O3D point cloud
        out_path (Path or str) Path were to save the point cloud to disk in ply format.

        Returns: None
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(path), self.pcd)
