from typing import List, Union, Tuple, TypedDict

from ..classes.camera import Camera
from ..classes.features import Features
from ..classes.point_cloud import PointCloud
from ..classes.images import ImageDS
from ..classes.targets import Targets

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


class PointCloudDict(TypedDict):
    epoch: PointCloud


class TargetDict(TypedDict):
    epoch: Targets
