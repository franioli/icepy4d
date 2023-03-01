import numpy as np
import pytest

from icepy.classes.point_cloud import PointCloud


def test_pointcloud_creation():
    points3d = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    points_col = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    example_pointcloud = PointCloud(points3d=points3d, points_col=points_col)
    assert len(example_pointcloud) == 3
    assert np.allclose(
        example_pointcloud.get_points(), np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    )
    assert np.allclose(
        example_pointcloud.get_colors(),
        np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
    )


def test_pointcloud_filter():
    points3d = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    points_col = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    example_pointcloud = PointCloud(points3d=points3d, points_col=points_col)
    example_pointcloud.sor_filter(nb_neighbors=1, std_ratio=1.0)
    assert len(example_pointcloud) == 0
