import numpy as np
import time
import pytest
import pandas as pd

from typing import TypedDict, List, Union
from scipy.spatial import Delaunay
from functools import wraps
from pathlib import Path

import icepy.classes as icepy_classes

from icepy.utils.spatial_funs import *


class FeaturesDictByCam(TypedDict):
    epoch: icepy_classes.Features


class FeaturesDictById(TypedDict):
    id: dict


class Features_tracked_inexes(TypedDict):
    fid: List[int]


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
def tracked_feautues_time_series(
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


@timeit
def tracked_points_time_series(
    points: icepy_classes.PointsDict,
    min_tracked_epoches: int = 1,
    volume: np.ndarray = None,
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

    epoches = list(points.keys())
    pts = {}
    for i, epoch in enumerate(epoches):
        track_ids = points[epoch].get_track_id_list()
        for track_id in track_ids:
            if volume is None:
                out = [ep for ep in epoches[i:] if track_id in points[ep]]
            else:
                out = [
                    ep
                    for ep in epoches[i:]
                    if (
                        track_id in points[ep]
                        and point_in_volume(points[ep][track_id].coordinates, volume)
                    )
                ]
            if not out:
                continue
            if len(out) < min_tracked_epoches:
                continue
            if track_id not in pts.keys():
                pts[track_id] = out
            else:
                pts[track_id].extend(x for x in out if x not in pts[track_id])

    return pts


@timeit
def tracked_dict_to_df(
    features: icepy_classes.FeaturesDict,
    points: icepy_classes.PointsDict,
    epoch_dict: icepy_classes.EpochDict,
    fts: Features_tracked_inexes,
    min_dt: int = None,
    vx_lims: List = None,
    vy_lims: List = None,
    vz_lims: List = None,
    save_path: Union[str, Path] = None,
) -> pd.DataFrame:
    """Convert dictionaries to a pandas DataFrame.

    Args:
        features (icepy_classes.FeaturesDict): A dictionary containing feature information.
        points (icepy_classes.PointsDict): A dictionary containing point information.
        epoch_dict (icepy_classes.EpochDict):
        fts (Features_tracked_inexes): A dictionary containing information about features that were tracked.
        min_dt (int, optional): The minimum number of days between `date_ini` and `date_fin`. Defaults to none.
        save_path (Union[str, Path], optional): The file path where the DataFrame should be saved. Defaults to None.

    Returns:
        pd.DataFrame: A pandas DataFrame containing information about the features and points.

    """
    cams = list(features[list(features.keys())[0]].keys())
    dict = {}
    ss = ["ini", "fin"]
    dict["fid"] = []
    dict["num_tracked_eps"] = []
    for s in ss:
        dict[f"ep_{s}"] = []
        dict[f"date_{s}"] = []
        for cam in cams:
            dict[f"x_{cam}_{s}"] = []
            dict[f"y_{cam}_{s}"] = []
        dict[f"X_{s}"] = []
        dict[f"Y_{s}"] = []
        dict[f"Z_{s}"] = []

    for fid in list(fts.keys()):
        dict["fid"].append(fid)
        dict["num_tracked_eps"].append(len(fts[fid]))
        eps = [fts[fid][0], fts[fid][-1]]
        for i, s in enumerate(ss):
            dict[f"ep_{s}"].append(eps[i])
            dict[f"date_{s}"].append(epoch_dict[eps[i]])
            for cam in cams:
                dict[f"x_{cam}_{s}"].append(features[eps[i]][cam][fid].x)
                dict[f"y_{cam}_{s}"].append(features[eps[i]][cam][fid].y)
            dict[f"X_{s}"].append(points[eps[i]][fid].X)
            dict[f"Y_{s}"].append(points[eps[i]][fid].Y)
            dict[f"Z_{s}"].append(points[eps[i]][fid].Z)

    fts_df = pd.DataFrame.from_dict(dict)
    fts_df["date_ini"] = pd.to_datetime(fts_df["date_ini"], format="%Y_%m_%d")
    fts_df["date_fin"] = pd.to_datetime(fts_df["date_fin"], format="%Y_%m_%d")
    fts_df["dt"] = pd.to_timedelta(fts_df["date_fin"] - fts_df["date_ini"], unit="D")
    fts_df["dX"] = fts_df["X_fin"] - fts_df["X_ini"]
    fts_df["dY"] = fts_df["Y_fin"] - fts_df["Y_ini"]
    fts_df["dZ"] = fts_df["Z_fin"] - fts_df["Z_ini"]
    fts_df["vX"] = fts_df["dX"] / fts_df["dt"].dt.days
    fts_df["vY"] = fts_df["dY"] / fts_df["dt"].dt.days
    fts_df["vZ"] = fts_df["dZ"] / fts_df["dt"].dt.days

    if min_dt is not None:
        fts_df = fts_df[fts_df["dt"] >= pd.to_timedelta(min_dt, unit="D")]

    if vx_lims is not None:
        keep = (fts_df["vX"] >= vx_lims[0]) & (fts_df["vX"] < vx_lims[1])
        fts_df = fts_df.loc[keep, :]
    if vy_lims is not None:
        keep = (fts_df["vY"] >= vy_lims[0]) & (fts_df["vY"] < vy_lims[1])
        fts_df = fts_df.loc[keep, :]
    if vz_lims is not None:
        keep = (fts_df["vZ"] >= vz_lims[0]) & (fts_df["vZ"] < vz_lims[1])
        fts_df = fts_df.loc[keep, :]

    if save_path is not None:
        fts_df.to_csv(save_path)

    return fts_df


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
