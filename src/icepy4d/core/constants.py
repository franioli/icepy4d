from typing import TypedDict

from .camera import Camera
from .features import Features
from .images import Image

DEFAULT_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


"""
Define basic data containers
"""


class FeaturesDict(TypedDict):
    camera: Features


class CamerasDict(TypedDict):
    camera: Camera


class ImagesDict(TypedDict):
    camera: Image


# class EpochDataMap:
#     def __init__(self, dict: dict[int, datetime]):
#         self._dict = dict

#     def __getitem__(self, key):
#         return str(self._dict[key]).replace(" ", "_")

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__} with {len(self._dict)} epochs"


# class FeaturesDictEpoch(TypedDict):
#     camera: Features


# class FeaturesDict(TypedDict):
#     epoch: FeaturesDictEpoch


# class CamerasDictEpoch(TypedDict):
#     camera: Camera


# class CamerasDict(TypedDict):
#     epoch: CamerasDictEpoch


# class PointsDict(TypedDict):
#     epoch: Points


# class PointCloudDict(TypedDict):
#     epoch: PointCloud


# class TargetDict(TypedDict):
#     epoch: Targets
