import numpy as np
import pandas as pd

from typing import TypedDict, List, Union
from pathlib import Path
from itertools import groupby, product

import icepy4d.classes as icepy4d_classes

from icepy4d.utils.spatial_funs import *
from icepy4d.utils.timer import timeit


class FeaturesDictByCam(TypedDict):
    epoch: icepy4d_classes.Features


class FeaturesDictById(TypedDict):
    id: dict


class Features_tracked_inexes(TypedDict):
    fid: List[int]


def sort_features_by_cam(
    features: icepy4d_classes.FeaturesDict, cam: str
) -> FeaturesDictByCam:
    """Sort features by camera.

    Args:
        features (icepy4d_classes.FeaturesDict): A dictionary of features, where the keys are epochs and the values are dictionaries of features for each camera.
        cam (str): The camera identifier to sort the features by.

    Returns:
        FeaturesDictByCam: A dictionary of features sorted by camera, where the keys are epochs and the values are the features for the specified camera.
    """
    f_by_cam: FeaturesDictByCam = {ep: features[ep][cam] for ep in features.keys()}
    return f_by_cam


# @timeit
# def tracked_features_time_series(
#     fdict: FeaturesDictByCam,
#     min_tracked_epoches: int = 1,
#     rect: np.ndarray = None,
# ) -> dict:
#     """
#     Calculates the time series of features that have been tracked.

#     Args:
#     fdict (FeaturesDictByCam): A dictionary containing features of each camera at different epochs.
#     min_tracked_epoches (int, optional): The minimum number of tracked epochs to be included in the time series. Defaults to 1.
#     rect (np.ndarray, optional): An optional rectangle used to filter the tracked features. Defaults to None.

#     Returns:
#     dict: A dictionary with track IDs as keys and the corresponding list of epochs in which the feature was tracked as values.

#     """
#     fts = {}
#     for track_id, group in groupby(
#         (
#             (ep, fid)
#             for ep, feats in fdict.items()
#             for fid, feat in feats.items()
#             if rect is None or point_in_rect(feat.xy.squeeze(), rect)
#         ),
#         key=lambda x: x[1].track_id,
#     ):
#         out = list(ep for ep, _ in group)
#         if len(out) >= min_tracked_epoches:
#             fts[track_id] = out
#     return fts


@timeit
def tracked_features_time_series(
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
        track_ids = fdict[epoch].get_track_ids()
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
    points: icepy4d_classes.PointsDict,
    min_tracked_epoches: int = 1,
    volume: np.ndarray = None,
) -> dict:
    """
    Calculates the time series of features that have been tracked.

    Args:
    points (PointsDict): A dictionary containing points at different epochs.
    min_tracked_epoches (int, optional): The minimum number of tracked epochs to be included in the time series. Defaults to 1.
    volume (np.ndarray, optional): An optional volume used to filter the tracked points. Defaults to None.

    Returns:
    dict: A dictionary with track IDs as keys and the corresponding list of epochs in which the point was tracked as values.

    """

    epoches = list(points.keys())

    # Get a list of tuples containing (track_id, epoch) for each point
    point_epochs = [
        (track_id, epoch)
        for epoch in epoches
        for track_id in points[epoch].get_track_ids()
    ]

    # Filter the points based on the given volume, if any
    if volume is not None:
        point_epochs = [
            (track_id, epoch)
            for track_id, epoch in point_epochs
            if point_in_volume(points[epoch][track_id].coordinates, volume)
        ]

    # Group the points by track ID
    point_groups = groupby(sorted(point_epochs), key=lambda x: x[0])

    # Construct the output dictionary
    pts = {}
    for track_id, point_group in point_groups:
        epoch_list = [epoch for _, epoch in point_group]
        if len(epoch_list) >= min_tracked_epoches:
            pts[track_id] = epoch_list

    return pts


# deprecated function (~10 times slower that new one)
def tracked_points_time_series_old(
    points: icepy4d_classes.PointsDict,
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
        track_ids = points[epoch].get_track_ids()
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
    features: icepy4d_classes.FeaturesDict,
    points: icepy4d_classes.PointsDict,
    epoch_dict: icepy4d_classes.EpochDict,
    fts: Features_tracked_inexes,
    min_dt: int = None,
    vx_lims: List = None,
    vy_lims: List = None,
    vz_lims: List = None,
    save_path: Union[str, Path] = None,
) -> pd.DataFrame:
    """Convert dictionaries to a pandas DataFrame.

    Args:
        features (icepy4d_classes.FeaturesDict): A dictionary containing feature information.
        points (icepy4d_classes.PointsDict): A dictionary containing point information.
        epoch_dict (icepy4d_classes.EpochDict):
        fts (Features_tracked_inexes): A dictionary containing information about features that were tracked.
        min_dt (int, optional): The minimum number of days between `date_ini` and `date_fin`. Defaults to none.
        save_path (Union[str, Path], optional): The file path where the DataFrame should be saved. Defaults to None.

    Returns:
        pd.DataFrame: A pandas DataFrame containing information about the features and points.

    """
    cams = list(features[list(features.keys())[0]].keys())
    dict = {
        "fid": [fid for fid in fts.keys()],
        "num_tracked_eps": [len(fts[fid]) for fid in fts.keys()],
        "ep_ini": [fts[fid][0] for fid in fts.keys()],
        "ep_fin": [fts[fid][-1] for fid in fts.keys()],
        "date_ini": [epoch_dict[fts[fid][0]] for fid in fts.keys()],
        "date_fin": [epoch_dict[fts[fid][-1]] for fid in fts.keys()],
    }
    for cam in cams:
        dict[f"x_{cam}_ini"] = [features[fts[fid][0]][cam][fid].x for fid in fts.keys()]
        dict[f"y_{cam}_ini"] = [features[fts[fid][0]][cam][fid].y for fid in fts.keys()]
        dict[f"x_{cam}_fin"] = [
            features[fts[fid][-1]][cam][fid].x for fid in fts.keys()
        ]
        dict[f"y_{cam}_fin"] = [
            features[fts[fid][-1]][cam][fid].y for fid in fts.keys()
        ]
    dict["X_ini"] = [points[fts[fid][0]][fid].X for fid in fts.keys()]
    dict["Y_ini"] = [points[fts[fid][0]][fid].Y for fid in fts.keys()]
    dict["Z_ini"] = [points[fts[fid][0]][fid].Z for fid in fts.keys()]
    dict["X_fin"] = [points[fts[fid][-1]][fid].X for fid in fts.keys()]
    dict["Y_fin"] = [points[fts[fid][-1]][fid].Y for fid in fts.keys()]
    dict["Z_fin"] = [points[fts[fid][-1]][fid].Z for fid in fts.keys()]

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
    fts_df["V"] = np.linalg.norm(fts_df[["vX", "vY", "vZ"]].to_numpy(), axis=1).reshape(
        -1, 1
    )

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


# @timeit
# def tracked_dict_to_df_old(
#     features: icepy4d_classes.FeaturesDict,
#     points: icepy4d_classes.PointsDict,
#     epoch_dict: icepy4d_classes.EpochDict,
#     fts: Features_tracked_inexes,
#     min_dt: int = None,
#     vx_lims: List = None,
#     vy_lims: List = None,
#     vz_lims: List = None,
#     save_path: Union[str, Path] = None,
# ) -> pd.DataFrame:
#     """Convert dictionaries to a pandas DataFrame.

#     Args:
#         features (icepy4d_classes.FeaturesDict): A dictionary containing feature information.
#         points (icepy4d_classes.PointsDict): A dictionary containing point information.
#         epoch_dict (icepy4d_classes.EpochDict):
#         fts (Features_tracked_inexes): A dictionary containing information about features that were tracked.
#         min_dt (int, optional): The minimum number of days between `date_ini` and `date_fin`. Defaults to none.
#         save_path (Union[str, Path], optional): The file path where the DataFrame should be saved. Defaults to None.

#     Returns:
#         pd.DataFrame: A pandas DataFrame containing information about the features and points.

#     """
#     cams = list(features[list(features.keys())[0]].keys())
#     dict = {}
#     ss = ["ini", "fin"]
#     dict["fid"] = []
#     dict["num_tracked_eps"] = []
#     for s in ss:
#         dict[f"ep_{s}"] = []
#         dict[f"date_{s}"] = []
#         for cam in cams:
#             dict[f"x_{cam}_{s}"] = []
#             dict[f"y_{cam}_{s}"] = []
#         dict[f"X_{s}"] = []
#         dict[f"Y_{s}"] = []
#         dict[f"Z_{s}"] = []

#     for fid in list(fts.keys()):
#         dict["fid"].append(fid)
#         dict["num_tracked_eps"].append(len(fts[fid]))
#         eps = [fts[fid][0], fts[fid][-1]]
#         for i, s in enumerate(ss):
#             dict[f"ep_{s}"].append(eps[i])
#             dict[f"date_{s}"].append(epoch_dict[eps[i]])
#             for cam in cams:
#                 dict[f"x_{cam}_{s}"].append(features[eps[i]][cam][fid].x)
#                 dict[f"y_{cam}_{s}"].append(features[eps[i]][cam][fid].y)
#             dict[f"X_{s}"].append(points[eps[i]][fid].X)
#             dict[f"Y_{s}"].append(points[eps[i]][fid].Y)
#             dict[f"Z_{s}"].append(points[eps[i]][fid].Z)

#     fts_df = pd.DataFrame.from_dict(dict)
#     fts_df["date_ini"] = pd.to_datetime(fts_df["date_ini"], format="%Y_%m_%d")
#     fts_df["date_fin"] = pd.to_datetime(fts_df["date_fin"], format="%Y_%m_%d")
#     fts_df["dt"] = pd.to_timedelta(fts_df["date_fin"] - fts_df["date_ini"], unit="D")
#     fts_df["dX"] = fts_df["X_fin"] - fts_df["X_ini"]
#     fts_df["dY"] = fts_df["Y_fin"] - fts_df["Y_ini"]
#     fts_df["dZ"] = fts_df["Z_fin"] - fts_df["Z_ini"]
#     fts_df["vX"] = fts_df["dX"] / fts_df["dt"].dt.days
#     fts_df["vY"] = fts_df["dY"] / fts_df["dt"].dt.days
#     fts_df["vZ"] = fts_df["dZ"] / fts_df["dt"].dt.days

#     if min_dt is not None:
#         fts_df = fts_df[fts_df["dt"] >= pd.to_timedelta(min_dt, unit="D")]

#     if vx_lims is not None:
#         keep = (fts_df["vX"] >= vx_lims[0]) & (fts_df["vX"] < vx_lims[1])
#         fts_df = fts_df.loc[keep, :]
#     if vy_lims is not None:
#         keep = (fts_df["vY"] >= vy_lims[0]) & (fts_df["vY"] < vy_lims[1])
#         fts_df = fts_df.loc[keep, :]
#     if vz_lims is not None:
#         keep = (fts_df["vZ"] >= vz_lims[0]) & (fts_df["vZ"] < vz_lims[1])
#         fts_df = fts_df.loc[keep, :]

#     if save_path is not None:
#         fts_df.to_csv(save_path)

#     return fts_df


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
