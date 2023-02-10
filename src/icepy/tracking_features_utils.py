import numpy as np
import time
import pytest

from typing import TypedDict
from scipy.spatial import Delaunay
from functools import wraps

import icepy.classes as icepy_classes

from icepy.utils.spatial import *


class FeaturesDictByCam(TypedDict):
    epoch: icepy_classes.Features


class FeaturesDictById(TypedDict):
    id: dict


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


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


@timeit
def tracked_time_series(
    fdict: FeaturesDictByCam,
    min_tracked_epoches: int = 1,
    rect: np.ndarray = None,
) -> dict:
    """
    Calculates the time series of features that have been tracked.

    Args:
    fdict (FeaturesDictByCam): A dictionary containing features of each camera at different epochs.
    min_tracked_epoches (int, optional): The minimum number of tracked epochs to be included in the time series. Defaults to 1.
    rect (np.ndarray, optional): An optional rectangle used to filter the tracked features. Defaults to None.

    Returns:
    dict: A dictionary with track IDs as keys and the corresponding list of epochs in which the feature was tracked as values.

    """

    epoches = list(fdict.keys())
    fts = {}
    for i, epoch in enumerate(epoches):
        track_ids = fdict[epoch].get_track_id_list()
        for track_id in track_ids:
            if rect is None:
                out = [ep for ep in epoches if track_id in fdict[ep]]
            else:
                out = [
                    ep
                    for ep in epoches[i:]
                    if (
                        track_id in fdict[ep]
                        and point_in_rect(fdict[ep][track_id].xy.squeeze(), rect)
                    )
                ]
            if not out:
                continue
            if len(out) < min_tracked_epoches:
                continue
            if track_id not in fts.keys():
                fts[track_id] = out
            else:
                fts[track_id].extend(x for x in out if x not in fts[track_id])

    return fts


# def tracked_time_series(
#     fdict: FeaturesDictByCam,
#     track_id: np.int32,
#     min_tracked_epoches: int = 1,
#     rect: np.ndarray = None,
# ) -> FeaturesDictById:
#     """Extract a time series of features for a specified track.

#     Args:
#         fdict (FeaturesDictByCam): A dictionary of features sorted by camera, where the keys are epochs and the values are Features object.
#         track_id (np.int32): The track identifier to extract the time series of features for.
#         min_tracked_epoches (int, optional): The minimum number of tracked epochs required for the time series to be returned. Defaults to 1.
#         rect (np.ndarray, optional): An optional bounding box to filter the features by. Defaults to `None`.

#     Returns:
#         FeaturesDict: A dictionary of features for the specified track, where the keys are epochs and the values are the features for the specified track. Returns `None` if the number of tracked epochs is less than `min_tracked_epoches` or if there are no features for the specified track.
#     """
#     epoches = list(fdict.keys())
#     if rect is None:
#         ts: FeaturesDictById = {
#             ep: fdict[ep][track_id] for ep in epoches if track_id in fdict[ep]
#         }
#     else:
#         ts: FeaturesDictById = {
#             ep: fdict[ep][track_id]
#             for ep in epoches
#             if track_id in fdict[ep] and feature_in_BB(fdict[ep][track_id], rect)
#         }
#     if not ts:
#         return None
#     if min_tracked_epoches > 0 and len(ts) <= min_tracked_epoches:
#         return None
#     return ts


if __name__ == "__main__":
    pass
