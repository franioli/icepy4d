from typing import TypedDict

from .camera import Camera
from .features import Features
from .images import Image


# Define basic data containers
class FeaturesDict(TypedDict):
    camera: Features


class CamerasDict(TypedDict):
    camera: Camera


class ImagesDict(TypedDict):
    camera: Image
