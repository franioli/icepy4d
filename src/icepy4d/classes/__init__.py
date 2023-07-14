from typing import Dict, Union, TypedDict

from .epoch import Epoch, Epoches
from .camera import Camera
from .images import Image, ImageDS
from .features import Feature, Features
from .point_cloud import PointCloud
from .targets import Targets
from .points import Point, Points
from .calibration import Calibration, read_opencv_calibration

from dataclasses import dataclass


# @dataclass
# class FeaturesDict:
#     camera: Features


# @dataclass
# class CamerasDict:
#     camera: Camera


class ImagesDict(TypedDict):
    camera: Image


# For backward compatibility. It must beintegrated in Epoches class
class EpochDict(TypedDict):
    epoch: str


class FeaturesDict(TypedDict):
    camera: Features


class CamerasDict(TypedDict):
    camera: Camera
