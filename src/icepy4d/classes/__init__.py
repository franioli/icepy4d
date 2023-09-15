from .base_containers import *  # noqa: F401
from .epoch import EpochDataMap, Epoch, Epoches  # noqa: F401
from .camera import Camera  # noqa: F401
from .images import Image, ImageDS  # noqa: F401
from .features import Feature, Features  # noqa: F401
from .point_cloud import PointCloud  # noqa: F401
from .targets import Targets  # noqa: F401
from .points import Point, Points  # noqa: F401
from .calibration import Calibration, read_opencv_calibration  # noqa: F401


# # For backward compatibility. It must beintegrated in Epoches class
# class EpochDataMap(TypedDict):
#     epoch: str


# class ImagesDict(TypedDict):
#     camera: Image


# class FeaturesDict(TypedDict):
#     camera: Features


# class CamerasDict(TypedDict):
#     camera: Camera
