import numpy as np

from typing import TypedDict
from scipy.spatial import Delaunay

# ICEpy4D
import icepy.classes as icepy_classes
import icepy.visualization as icepy_viz


class FeaturesDictByCam(TypedDict):
    epoch: icepy_classes.Features


class FeaturesDictById(TypedDict):
    id: dict


def sort_features_by_cam(
    features: icepy_classes.FeaturesDict, cam: str
) -> FeaturesDictByCam:
    """Sort features by camera.

    Args:
        features (icepy_classes.FeaturesDict): A dictionary of features, where the keys are epochs and the values are dictionaries of features for each camera.
        cam (str): The camera identifier to sort the features by.

    Returns:
        FeaturesDictByCam: A dictionary of features sorted by camera, where the keys are epochs and the values are the features for the specified camera.
    """
    f_by_cam: FeaturesDictByCam = {ep: features[ep][cam] for ep in features.keys()}
    return f_by_cam


def tracked_time_series(
    fdict: FeaturesDictByCam,
    track_id: np.int32,
    min_tracked_epoches: int = 1,
    rect: np.ndarray = None,
) -> FeaturesDictById:
    """Extract a time series of features for a specified track.

    Args:
        fdict (FeaturesDict): A dictionary of features sorted by camera, where the keys are epochs and the values are Features object.
        track_id (np.int32): The track identifier to extract the time series of features for.
        min_tracked_epoches (int, optional): The minimum number of tracked epochs required for the time series to be returned. Defaults to 1.
        rect (np.ndarray, optional): An optional bounding box to filter the features by. Defaults to `None`.

    Returns:
        FeaturesDict: A dictionary of features for the specified track, where the keys are epochs and the values are the features for the specified track. Returns `None` if the number of tracked epochs is less than `min_tracked_epoches` or if there are no features for the specified track.
    """
    epoches = list(fdict.keys())
    if rect is None:
        ts: FeaturesDictById = {
            ep: fdict[ep][track_id] for ep in epoches if track_id in fdict[ep]
        }
    else:
        ts: FeaturesDictById = {
            ep: fdict[ep][track_id]
            for ep in epoches
            if track_id in fdict[ep] and feature_in_BB(fdict[ep][track_id], rect)
        }
    if not ts:
        return None
    if min_tracked_epoches > 0 and len(ts) <= min_tracked_epoches:
        return None
    return ts


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


def feature_in_BB(feature: icepy_classes.Feature, rect: np.array) -> bool:
    """
    feature_in_BB check if a feature is within a given bounding box

    Args:
        feature (icepy_classes.Feature): Feature object
        rect (np.array): numpy array with bounding box coordinates as [xmin, ymin, xmax, ymax]

    Returns:
        bool: true or false
    """
    pt = feature.xy.squeeze()
    return point_in_rect(pt, rect)


def point_in_volume(point: icepy_classes.Point, coordinates: np.array) -> bool:
    """
    point_in_volume _summary_

    Args:
        point (icepy_classes.Point): _description_
        coordinates (np.array): _description_

    Returns:
        bool: _description_
    """
    pass
