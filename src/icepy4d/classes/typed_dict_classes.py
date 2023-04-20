from typing import List, Union, Tuple, TypedDict

from .camera import Camera
from .features import Features
from .point_cloud import PointCloud
from .images import ImageDS
from .targets import Targets
from .points import Points

"""
Define TypedDict classes for storing data from all the epoches
"""


class FeaturesDictEpoch(TypedDict):
    camera: Features


class FeaturesDict(TypedDict):
    epoch: FeaturesDictEpoch


class CamerasDictEpoch(TypedDict):
    camera: Camera


class CamerasDict(TypedDict):
    epoch: CamerasDictEpoch


class ImagesDict(TypedDict):
    camera: ImageDS


class EpochDict(TypedDict):
    epoch: str


class PointsDict(TypedDict):
    epoch: Points


class PointCloudDict(TypedDict):
    epoch: PointCloud


class TargetDict(TypedDict):
    epoch: Targets
