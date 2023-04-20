import numpy as np
import pytest

from pathlib import Path

from icepy4d.classes.point_cloud import PointCloud


@pytest.fixture
def example_pointcloud():
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return PointCloud(points3d=points, points_col=colors)


def test_create_from_numpy(example_pointcloud):
    assert len(example_pointcloud) == 3
    assert np.allclose(
        example_pointcloud.get_points(), np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    )
    assert np.allclose(
        example_pointcloud.get_colors(),
        np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
    )


def test_create_from_numpy():
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pcd = PointCloud(points3d=points, points_col=colors)
    assert len(pcd) == 3


def test_pointcloud_creation(example_pointcloud):
    assert len(example_pointcloud) == 3
    assert np.allclose(
        example_pointcloud.get_points(), np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    )
    assert np.allclose(
        example_pointcloud.get_colors(),
        np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
    )


def test_sor_filter(example_pointcloud):
    example_pointcloud.sor_filter(nb_neighbors=1, std_ratio=1.0)
    assert len(example_pointcloud) == 0


# def test_read_las(example_pointcloud):
#     las_file = Path("test_data/las_file.las")
#     pointcloud = PointCloud(pcd_path=las_file)
#     assert len(pointcloud) == 5 # or whatever the number of points in the las file is
#     # assert that the point coordinates and colors are correctly stored in the point cloud object

# def test_write_and_read_ply(example_pointcloud, tmp_path):
#     ply_file = tmp_path / "test.ply"
#     example_pointcloud.write_ply(ply_file)
#     pointcloud = PointCloud(pcd_path=ply_file)
#     assert len(pointcloud) == 3
#     # assert that the point coordinates and colors are correctly stored in the point cloud object
