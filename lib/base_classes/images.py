"""
MIT License

Copyright (c) 2022 Francesco Ioli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import logging
from pathlib import Path
from typing import List, Union
from datetime import datetime

import cv2
import exifread
import numpy as np

# from lib.utils.sensor_width_database import SensorWidthDatabase


def read_image(
    path: Union[str, Path],
    color: bool = True,
    resize: List[int] = [-1],
    crop: List[int] = None,
) -> np.ndarray:
    """
    Read image with OpenCV and return it as np array
    __________
    Parameters:
    - path (str or Path): image path
    - color (bool, default=True): read color image or grayscale
    - resize (List, default=[-1]): If not [-1], image is resized at [w, h] dimensions
    - crop (List, default=None): If not None, List containing bounding box for cropping the image as [xmin, xmax, ymin, ymax]
    __________
    Return
        image (np.ndarray): image
    """

    if color:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE

    try:
        image = cv2.imread(str(path), flag)
    except:
        print(f"Impossible to load image {path}")

    if color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        if len(resize) == 1 and resize[0] == -1:
            return None
        else:
            return None, None

    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    image = cv2.resize(image, (w_new, h_new))
    if crop:
        image = image[crop[1] : crop[3], crop[0] : crop[2]]

    if len(resize) == 1 and resize[0] == -1:
        return image
    else:
        return image, scales


def process_resize(w, h, resize):
    assert len(resize) > 0 and len(resize) <= 2
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]
    return w_new, h_new


class Image:
    def __init__(self, path: Union[str, Path], image: np.ndarray = None) -> None:

        self._path = path
        self._value_array = None
        self._width = None
        self._height = None
        self._exif_data = None
        self._date_time = None
        self.read_exif()

        if image:
            self._value_array = image

    @property
    def height(self) -> int:
        """
        The height of the image (i.e. number of pixels in the vertical direction).
        """
        if self._height:
            return int(self._height)

    @property
    def width(self) -> int:
        """
        The width of the image (i.e. number of pixels in the horizontal direction).
        """
        if self._width:
            return int(self._width)

    @property
    def path(self) -> str:
        """
        Path of the image
        """
        return self._path

    @property
    def exif(self) -> dict:
        return self._exif_data

    @property
    def date(self):
        return self._date_time

    def get_image(self) -> np.ndarray:
        """Returns the image"""
        if self._value_array is not None:
            return self._value_array
        else:
            return self.read_image(self._path)

    def read_image(
        self,
        path: Union[str, Path],
        col: bool = True,
        resize: List[int] = [-1],
        crop: List[int] = None,
    ) -> None:
        """Wrapper around the function read_image to be a class method."""
        path = Path(path)
        if path.exists():
            self._value_array = read_image(path, col, resize, crop)
            self.read_exif()
        else:
            print(f"Input paht {path} not valid.")

    def clean_image(self) -> None:
        self._value_array = None

    def read_exif(self) -> None:
        """Read image exif with exifread and store them in a dictionary"""
        try:
            f = open(self._path, "rb")
            self._exif_data = exifread.process_file(f, details=False)
            f.close()
        except:
            print("No exif data available.")

        # Set image size
        if (
            "Image ImageWidth" in self._exif_data.keys()
            and "Image ImageLength" in self._exif_data.keys()
        ):
            self._width = self._exif_data["Image ImageWidth"].printable
            self._height = self._exif_data["Image ImageLength"].printable

        # Set Image Datetime
        if "Image DateTime" in self._exif_data.keys():
            date_fmt = "%Y:%m:%d %H:%M:%S"
            date_str = self._exif_data["Image DateTime"].printable
            self._date_time = datetime.strptime(date_str, date_fmt)

    def extract_patch(self, limits: dict) -> np.ndarray:
        """Extract image patch
        Parameters
        __________
        - limits (dict): dictionary containing the index of the tile (in row-major order, C-style) and a list of the bounding box coordinates as: {0,[xmin, xmax, ymin, ymax]}
        __________
        Return: patch (np.ndarray)
        """
        image = read_image(self._path)
        patch = image[
            limits[1] : limits[3],
            limits[0] : limits[2],
        ]
        return patch

    def get_intrinsics_from_exif(self) -> None:
        """Constructs the camera intrinsics from exif tag.

        Equation: focal_px=max(w_px,h_px)∗focal_mm / ccdw_mm

        Ref:
        - https://github.com/colmap/colmap/blob/e3948b2098b73ae080b97901c3a1f9065b976a45/src/util/bitmap.cc#L282
        - https://openmvg.readthedocs.io/en/latest/software/SfM/SfMInit_ImageListing/
        - https://photo.stackexchange.com/questions/40865/how-can-i-get-the-image-sensor-dimensions-in-mm-to-get-circle-of-confusion-from # noqa: E501

        Returns:
            intrinsics matrix (3x3).
        """

        # if self._exif_data is None or len(self._exif_data) == 0:
        #     return None

        # focal_length_mm = self.exif_data.get("FocalLength")

        # sensor_width_mm = Image.sensor_width_db.lookup(
        #     self._exif_data.get("Make"),
        #     self._exif_data.get("Model"),
        # )

        # img_w_px = self._width
        # img_h_px = self._height
        # focal_length_px = max(img_h_px, img_w_px) * \
        #     focal_length_mm / sensor_width_mm

        # center_x = img_w_px / 2
        # center_y = img_h_px / 2


class Imageds:
    """
    Class to help manage Image datasets

    """

    def __init__(
        self,
        path=None,
        logger: logging = None,
    ):
        # TODO: implement labels in datastore
        if not hasattr(self, "files"):
            self.reset_imageds()
        if path is not None:
            self.get_image_list(path)

    def __len__(self):
        """Get number of images in the datastore"""
        return len(self.files)

    def __contains__(self, name):
        """Check if an image is in the datastore, given the image name"""
        return name in self.files

    def __getitem__(self, idx, **args):
        """Read and return the image at position idx in the image datastore"""
        # TODO: add possibility to chose reading between col or grayscale, scale image, crop etc...
        # @TODO change getitem to return path and implement reading function
    
        img = read_image(os.path.join(self.folder[idx], self.files[idx]))
        assert img is not None, f"Unable to read image {self.files[idx]}"
    
        return img

    def reset_imageds(self):
        """Initialize image datastore"""
        self.files = []
        self.folder = []
        self.ext = []
        self.label = []
        # self.size = []
        # self.shot_date = []
        # self.shot_time = []

    def get_image_list(self, path):
        # TODO: change name in read image list
        # TODO: add option for including subfolders
        if not os.path.exists(path):
            print("Error: invalid input path.")
            return
        d = os.listdir(path)
        d.sort()
        self.files = d
        self.folder = [path] * len(d)

    def get_image_name(self, idx):
        """Return image name at position idx in datastore"""
        return self.files[idx]

    def get_image_path(self, idx):
        """Return full path of the image at position idx in datastore"""
        return os.path.join(self.folder[idx], self.files[idx])

    def get_image_stem(self, idx):
        """Return name without extension(stem) of the image at position idx in datastore"""
        return Path(self.files[idx]).stem


if __name__ == "__main__":
    """Test classes"""

    cams = ["p1", "p2"]
    images = dict.fromkeys(cams)
    for cam in cams:
        images[cam] = Imageds(Path("data/img2022") / cam)

    im = Image(images["p1"].get_image_path(0))
    print(im)
