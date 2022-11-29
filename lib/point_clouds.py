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

from pathlib import Path

from lib.utils.utils import create_directory


def create_point_cloud(points3d, points_col=None, path=None, *scalar_fied):
    """Function to create a point cloud object by using Open3D library.
    Input:  (nx3, float32) array of points 3D.
            (nx3, float32) array of color of each point.
                Colors are defined in [0,1] range as float numbers.
            Path were to save the point cloud to disk in ply format.
                If path is None, the point cloud is not saved to disk.
            Scalar fields: to be implemented.
            #TODO: implement scalar fields.
    Return: Open3D point cloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    if points_col is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_col)
    if path is not None:
        write_ply(pcd, path)
    return pcd


def write_ply(pcd, path):
    """Write point cloud to disk as .ply

    Parameters
    ----------
    pcd : O3D point cloud
    out_path : Path or str of output ply

    Returns: None
    """
    path = Path(path)
    create_directory(path.parent)
    o3d.io.write_point_cloud(str(path), pcd)
