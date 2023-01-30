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

from ..utils.sensor_width_database import SensorWidthDatabase


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
        logging.error(f"Impossible to load image {path}")

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
        """
        __init__ Create Image object

        Args:
            path (Union[str, Path]): path to the image
            image (np.ndarray, optional): Numpy array containing pixel values. If provided, they are stored in self._value_array and they are accessible from outside the class with Image.value. Defaults to None.
        """

        self._path = Path(path)
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
        else:
            logging.error("Image height not available. Read it from exif first")
            return None

    @property
    def width(self) -> int:
        """
        The width of the image (i.e. number of pixels in the horizontal direction).
        """
        if self._width:
            return int(self._width)
        else:
            logging.error("Image width not available. Read it from exif first")
            return None

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
    def date(self) -> str:
        """
        date Returns the date of the image from exif as a string
        """
        if self._date_time is not None:
            return self._date_time.strftime("%Y:%m:%d")
        else:
            logging.error("No exif data available.")

    @property
    def time(self) -> str:
        """
        time Returns the time of the image from exif as a string

        """
        if self._date_time is not None:
            return self._date_time.strftime("%H:%M:%S")
        else:
            logging.error("No exif data available.")

    @property
    def value(self) -> np.ndarray:
        """
        Returns the image (pixel values) as numpy array
        """
        if self._value_array is not None:
            return self._value_array
        else:
            return self.read_image(self._path)

    def get_datetime(self):
        return self._date_time

    def read_image(
        self,
        # path: Union[str, Path],
        col: bool = True,
        resize: List[int] = [-1],
        crop: List[int] = None,
    ) -> None:
        """Wrapper around the function read_image to be a class method."""
        # path = Path(path)
        if self.path.exists():
            self._value_array = read_image(self.path, col, resize, crop)
            self.read_exif()
        else:
            logging.error(f"Input paht {self.path} not valid.")

    def clean_image(self) -> None:
        self._value_array = None

    def read_exif(self) -> None:
        """Read image exif with exifread and store them in a dictionary"""
        try:
            f = open(self._path, "rb")
            self._exif_data = exifread.process_file(f, details=False)
            f.close()
        except:
            logging.error("No exif data available.")

        # Get image size
        if (
            "Image ImageWidth" in self._exif_data.keys()
            and "Image ImageLength" in self._exif_data.keys()
        ):
            self._width = self._exif_data["Image ImageWidth"].printable
            self._height = self._exif_data["Image ImageLength"].printable
        elif (
            "EXIF ExifImageWidth" in self._exif_data.keys()
            and "EXIF ExifImageLength" in self._exif_data.keys()
        ):
            self._width = self._exif_data["EXIF ExifImageWidth"].printable
            self._height = self._exif_data["EXIF ExifImageLength"].printable
        else:
            logging.error(
                "Image width and height found in exif. Try to load the image and get image size from numpy array"
            )
            try:
                img = Image(self.path)
                self.height, self.width = img.height, img.width

            except:
                raise RuntimeError("Unable to get image dimensions.")

        # Get Image Date and Time
        self._date_time_fmt = "%Y:%m:%d %H:%M:%S"
        if "Image DateTime" in self._exif_data.keys():
            date_str = self._exif_data["Image DateTime"].printable
        elif "EXIF DateTimeOriginal" in self._exif_data.keys():
            date_str = self._exif_data["EXIF DateTimeOriginal"].printable
        else:
            logging.error("Date not available in exif.")
            return
        self._date_time = datetime.strptime(date_str, self._date_time_fmt)

    def extract_patch(self, limits: List[int]) -> np.ndarray:
        """Extract image patch
        Parameters
        __________
        - limits (List[int]): List containing the bounding box coordinates as: [xmin, xmax, ymin, ymax]
        __________
        Return: patch (np.ndarray)
        """
        image = read_image(self._path)
        patch = image[
            limits[1] : limits[3],
            limits[0] : limits[2],
        ]
        return patch

    def get_intrinsics_from_exif(self) -> np.ndarray:
        """Constructs the camera intrinsics from exif tag.

        Equation: focal_px=max(w_px,h_px)*focal_mm / ccdw_mm

        Ref:
        - https://github.com/colmap/colmap/blob/e3948b2098b73ae080b97901c3a1f9065b976a45/src/util/bitmap.cc#L282
        - https://openmvg.readthedocs.io/en/latest/software/SfM/SfMInit_ImageListing/
        - https://photo.stackexchange.com/questions/40865/how-can-i-get-the-image-sensor-dimensions-in-mm-to-get-circle-of-confusion-from # noqa: E501

        Returns:
            K (np.ndarray): intrinsics matrix (3x3 numpy array).
        """
        if self._exif_data is None or len(self._exif_data) == 0:
            try:
                self.read_exif()
            except OSError:
                logging.error("Unable to read exif data.")
                return None
        try:
            focal_length_mm = float(self._exif_data["EXIF FocalLength"].printable)
        except OSError:
            logging.error("Focal length non found in exif data.")
            return None
        try:
            sensor_width_db = SensorWidthDatabase()
            sensor_width_mm = sensor_width_db.lookup(
                self._exif_data["Image Make"].printable,
                self._exif_data["Image Model"].printable,
            )
        except OSError:
            logging.error("Unable to get sensor size in mm from sensor database")
            return None

        img_w_px = self.width
        img_h_px = self.height
        focal_length_px = max(img_h_px, img_w_px) * focal_length_mm / sensor_width_mm
        center_x = img_w_px / 2
        center_y = img_h_px / 2
        K = np.array(
            [
                [focal_length_px, 0.0, center_x],
                [0.0, focal_length_px, center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return K


class ImageDS:
    """
    Class to manage Image datasets for multi epoch

    """

    def __init__(
        self,
        folder: Union[str, Path],
        ext: str = None,
        recursive: bool = False,
    ) -> None:
        """
        __init__ _summary_

        Args:
            folder (Union[str, Path]): Path to the image folder
            ext (str, optional): Image extension for filtering files. If None is provided, all files in 'folder' are read. Defaults to None.
            recursive (bool, optional): Read files recurevely. Defaults to False.

        Raises:
            IsADirectoryError: _description_
        """
        self.reset_imageds()

        self.folder = Path(folder)
        if not self.folder.exists():
            msg = f"Error: invalid input path {self.folder}"
            logging.error(msg)
            raise IsADirectoryError(msg)
        if ext is not None:
            self.ext = ext
        self.recursive = recursive

        self.read_image_list(self.folder)

    def __len__(self) -> int:
        """Get number of images in the datastore"""
        return len(self.files)

    def __contains__(self, name: str) -> bool:
        """Check if an image is in the datastore, given the image name"""
        files = [x.name for x in self.files]
        return name in files

    def __getitem__(self, idx: int) -> str:
        """Return image name (including extension) at position idx in datastore"""
        return self.files[idx].name

    def __iter__(self):
        self._elem = 0
        return self

    def __next__(self):
        while self._elem < len(self):
            file = self.files[self._elem]
            self._elem += 1
            return file
        else:
            self._elem
            raise StopIteration

    def reset_imageds(self) -> None:
        """Initialize image datastore"""
        self.files = None
        self.folder = None
        self.ext = None
        self._elem = 0

    def read_image_list(self, recursive: bool = None) -> None:
        assert self.folder.is_dir(), "Error: invalid image directory."

        if recursive is not None:
            self.recursive = recursive
        if self.recursive:
            rec_patt = "**/"
        else:
            rec_patt = ""
        if self.ext is not None:
            ext_patt = f".{self.ext}"
        else:
            ext_patt = ""
        pattern = f"{rec_patt}*{ext_patt}"

        self.files = sorted(self.folder.glob(pattern))

        if len(self.files) == 0:
            logging.error(f"No images found in folder {self.folder}")
            return
        try:
            self.read_dates()
        except OSError as err:
            logging.exception(err)

    def read_image(self, idx: int) -> Image:
        """Return image at position idx as Image instance, containing both exif and value data (accessible by value proprierty, e.g., image.value)"""
        image = Image(self.files[idx])
        image.read_image()
        return image

    def read_dates(self) -> None:
        """
        read_dates Read date and time for all the images in ImageDS from exif.
        """
        assert self.files, "No image in ImageDS. Please read image list first"
        self._dates, self._times = {}, {}
        try:
            for id, im in enumerate(self.files):
                image = Image(im)
                self._dates[id] = image.date
                self._times[id] = image.time
        except:
            logging.error("Unable to read image dates and time from exif.")
            self._dates, self._times = {}, {}
            return

    def get_image_path(self, idx: int) -> Path:
        """Return path of the image at position idx in datastore as Pathlib"""
        return self.files[idx]

    def get_image_stem(self, idx: int) -> str:
        """Return name without extension(stem) of the image at position idx in datastore"""
        return self.files[idx].stem

    def get_image_date(self, idx: int) -> str:
        """Return name without extension(stem) of the image at position idx in datastore"""
        return self._dates[idx]

    def get_image_time(self, idx: int) -> str:
        """Return name without extension(stem) of the image at position idx in datastore"""
        return self._times[idx]

    def write_exif_to_csv(
        self, filename: str, sep: str = ",", header: bool = True
    ) -> None:
        assert self.folder.is_dir(), "Empty Image Datastore."
        file = open(filename, "w")
        if header:
            file.write("epoch,name,date,time\n")
        for i, img_path in enumerate(self.files):
            img = Image(img_path)
            name = img_path.name
            date = img.date
            time = img.time
            file.write(f"{i}{sep}{name}{sep}{date}{sep}{time}\n")
        file.close()


if __name__ == "__main__":
    """Test classes"""

    images = ImageDS("data/img2022/p1")

    # Read image dates and times
    # images.read_dates()
    print(images.get_image_date(0))
    print(images.get_image_time(0))

    # Get image name
    print(images[0])

    # Get image stem
    print(images.get_image_stem(0))

    # Get image path
    print(images.get_image_path(0))

    # Get image as Image object and extect date and time
    img = images.read_image(0)
    print(img.date)
    print(img.time)

    # Read image as numpy array
    image = images.read_image(0).value

    # Test ImageDS iterator
    print(next(images))
    print(next(images))
    for i in images:
        print(i)

    # Build intrinics from exif
    image = images.read_image(0)
    K = image.get_intrinsics_from_exif()
    print(K)

    # Write exif to csv file
    # filename = "test.csv"
    # images.write_exif_to_csv(filename)

    # cams = ["p1", "p2"]
    # images = {}
    # for cam in cams:
    #     images[cam] = ImageDS(Path("data/img2021") / cam)
    #     images[cam].write_exif_to_csv(f"data/img2021/image_list_{cam}.csv")

    print("Done")
