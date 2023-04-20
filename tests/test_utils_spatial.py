import numpy as np

from icepy4d.utils.spatial_funs import *


def test_point_in_volume():
    volume = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    point = (0.5, 0.5, 0.5)
    assert point_in_volume(point, volume) == True


def test_point_not_in_volume():
    volume = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    point = (2, 2, 2)
    assert point_in_volume(point, volume) == False


def test_ccw_sort_points():
    points = np.array([[1, 2], [2, 3], [1, 3], [2, 2]])
    sorted_points = ccw_sort_points(points)
    expected_sorted_points = np.array([[1, 2], [1, 3], [2, 3], [2, 2]])
    np.testing.assert_array_equal(sorted_points, expected_sorted_points)


def test_ccw_sort_points_with_colinear_points():
    points = np.array([[1, 2], [2, 2], [3, 2]])
    sorted_points = ccw_sort_points(points)
    expected_sorted_points = np.array([[1, 2], [2, 2], [3, 2]])
    np.testing.assert_array_equal(sorted_points, expected_sorted_points)


def test_point_in_hull():
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    hull = Delaunay(points)
    p = np.array([[0.5, 0.5]])
    assert point_in_hull(p, hull) == True
    p = np.array([[2, 2]])
    assert point_in_hull(p, hull) == False


def test_point_in_rect():
    point = np.array([1, 1])
    rect = np.array([0, 0, 2, 2])
    assert point_in_rect(point, rect) == True
    point = np.array([3, 3])
    assert point_in_rect(point, rect) == False
