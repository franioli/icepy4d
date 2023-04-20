import numpy as np

from scipy.spatial import Delaunay

from icepy4d.classes.features import Features, Feature
from icepy4d.classes.points import Point


def ccw_sort_points(p: np.ndarray) -> np.ndarray:
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


def point_in_hull(p, hull):
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


def point_in_rect(point: np.ndarray, rect: np.array) -> bool:
    """
    point_in_rect check if a point is within a given bounding box

    Args:
        point (np.ndarray): numpy array with 2D x,y coordinates
        rect (np.array): numpy array with bounding box coordinates as [xmin, ymin, xmax, ymax]

    Returns:
        bool: true or false
    """
    logic = rect[0] < point[0] < rect[2] and rect[1] < point[1] < rect[3]
    return logic


def points_in_rect(points: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """
    points_in_rect checks which points are within a given bounding box

    Args:
        points (np.ndarray): numpy array with 2D x,y coordinates of shape (n,2)
        rect (np.array): numpy array with bounding box coordinates as [xmin, ymin, xmax, ymax]

    Returns:
        np.ndarray: boolean numpy array of shape (n,) indicating whether each point is within the bounding box
    """
    logic = np.all(points > rect[:2], axis=1) & np.all(points < rect[2:], axis=1)
    return logic


def feature_in_rect(feature: Feature, rect: np.array) -> bool:
    """Wrapper around point_in_rect function to deal with Feature object"""
    pt = feature.xy.squeeze()
    return point_in_rect(pt, rect)


def select_features_by_rect(features: Features, rect: np.ndarray) -> Features:
    pts = features.to_numpy()["kpts"]
    track_id_list = features.get_track_ids()
    valid = points_in_rect(pts, rect)

    selected = Features()
    selected.append_features_from_numpy(
        x=pts[valid, 0],
        y=pts[valid, 1],
        track_ids=list(np.array(track_id_list)[valid]),
    )

    return selected


def point_in_volume(point: np.ndarray, volume: np.array) -> bool:
    """
    Check if a point is within a cubic volume.

    Parameters
    ----------
    point : np.ndarray
        A 3D point given as a tuple of x, y, and z coordinates.
    volume : numpy.ndarray
        A 3D cubic volume defined as a numpy array with 3D coordinates.

    Returns
    -------
    bool
        True if the point is within the volume, False otherwise.

    Example
    -------
    >>> volume = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> point = (0.5, 0.5, 0.5)
    >>> is_point_in_volume(point, volume)
    True
    """
    x, y, z = point
    x_min, x_max = np.min(volume[:, 0]), np.max(volume[:, 0])
    y_min, y_max = np.min(volume[:, 1]), np.max(volume[:, 1])
    z_min, z_max = np.min(volume[:, 2]), np.max(volume[:, 2])
    return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max


def point3D_in_volume(point3D: Point, volume: np.array) -> bool:
    """Wrapper around point_in_rect function to deal with Point object"""
    pt = point3D.coordinates.squeeze()
    return point_in_volume(pt, volume)
