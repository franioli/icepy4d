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

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Union, Dict

import cv2
import exifread
import numpy as np


from .camera import Camera
from .sensor_width_database import SensorWidthDatabase
from .constants import DATE_FMT, DATETIME_FMT, TIME_FMT

logger = logging.getLogger(__name__)
logging.getLogger("exifread").setLevel(logging.WARNING)


# @TODO: remove variable number of outputs
def read_image(
    path: Union[str, Path],
    color: bool = True,
    resize: List[int] = [-1],
    crop: List[int] = None,
) -> np.ndarray:
    """
    Reads image with OpenCV and returns it as a NumPy array.

    Args:
        path (Union[str, Path]): The path of the image.
        color (bool, optional): Whether to read the image as color (RGB) or grayscale. Defaults to True.
        resize (List[int], optional): If not [-1], image is resized at [width, height] dimensions. Defaults to [-1].
        crop (List[int], optional): If not None, a List containing the bounding box for cropping the image as [xmin, xmax, ymin, ymax]. Defaults to None.

    Returns:
        np.ndarray: The image as a NumPy array.
    """

    if color:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE

    try:
        image = cv2.imread(str(path), flag)
    except:
        logger.error(f"Impossible to load image {path}")
        return None, None

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
    """A class representing an image.

    Attributes:
        _path (Path): The path to the image file.
        _value_array (np.ndarray): Numpy array containing pixel values. If available, it can be accessed with `Image.value`.
        _width (int): The width of the image in pixels.
        _height (int): The height of the image in pixels.
        _exif_data (dict): The EXIF metadata of the image, if available.
        _date_time (datetime): The date and time the image was taken, if available.

    """

    def __init__(self, path: Union[str, Path]) -> None:
        """
        __init__ Create Image object as a lazy loader for image data

        Args:
            path (Union[str, Path]): path to the image
        """

        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Invalid input path. Image {self._path} not found.")
        
        self._value_array = None
        self._width = None
        self._height = None
        self._exif_data = None
        self._date_time = None
        self.read_exif()

    def __repr__(self) -> str:
        """Returns a string representation of the image"""
        return f"Image {self._path}"

    @property
    def height(self) -> int:
        """Returns the height of the image"""
        if self._height:
            return int(self._height)
        else:
            logger.error("Image height not available. Read it from exif first")
            return None

    @property
    def width(self) -> int:
        """Returns the width of the image"""
        if self._width:
            return int(self._width)
        else:
            logger.error("Image width not available. Read it from exif first")
            return None

    @property
    def name(self) -> str:
        """Returns the name of the image (including extension)"""
        return self._path.name

    @property
    def stem(self) -> str:
        """Returns the name of the image (excluding extension)"""
        return self._path.stem

    @property
    def path(self) -> str:
        """Path of the image"""
        return self._path

    @property
    def parent(self) -> str:
        """Path to the parent folder of the image"""
        return self._path.parent

    @property
    def extension(self) -> str:
        """Returns the extension  of the image"""
        return self._path.suffix

    @property
    def exif(self) -> dict:
        """
        exif Returns the exif of the image

        Returns:
            dict: Dictionary containing Exif information
        """
        return self._exif_data

    @property
    def date(self) -> str:
        """
        Returns the date and time of the image in a string format.

        Returns:
            str: The date and time of the image.
        """
        if self._date_time is not None:
            return self._date_time.strftime(DATE_FMT)
        else:
            logger.error("No exif data available.")
            return

    @property
    def time(self) -> str:
        """
        time Returns the time of the image from exif as a string

        """
        if self._date_time is not None:
            return self._date_time.strftime(TIME_FMT)
        else:
            logger.error("No exif data available.")
            return None

    @property
    def datetime(self) -> datetime:
        """
        Returns the date and time of the image as datetime object.

        Returns:
            datetime: The date and time of the image as datetime object
        """
        if self._date_time is not None:
            return self._date_time
        else:
            logger.error("No exif data available.")
            return

    @property
    def timestamp(self) -> str:
        """
        Returns the date and time of the image in a string format.

        Returns:
            str: The date and time of the image as datetime object
        """
        if self._date_time is not None:
            return self._date_time.strftime(DATETIME_FMT)
        else:
            logger.error("No exif data available.")
            return

    @property
    def value(self) -> np.ndarray:
        """
        Returns the image (pixel values) as numpy array
        """
        if self._value_array is not None:
            return self._value_array
        else:
            return self.read_image(self._path)

    def read_image(
        self,
        # path: Union[str, Path],
        col: bool = True,
        resize: List[int] = [-1],
        crop: List[int] = None,
    ) -> np.ndarray:
        """Wrapper around the function read_image to be a class method."""
        # path = Path(path)
        if self.path.exists():
            self._value_array = read_image(self.path, col, resize, crop)
            self.read_exif()
            return self._value_array
        else:
            logger.error(f"Input paht {self.path} not valid.")
            return None

    def reset_image(self) -> None:
        self._value_array = None

    def read_exif(self) -> None:
        """Read image exif with exifread and store them in a dictionary"""
        try:
            with open(self._path, "rb") as f:
                self._exif_data = exifread.process_file(f, details=False, debug=False)
        except OSError:
            logger.error("No exif data available.")

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
            logger.error(
                "Image width and height found in exif. Try to load the image and get image size from numpy array"
            )
            try:
                img = Image(self.path)
                self.height, self.width = img.height, img.width

            except OSError:
                raise RuntimeError("Unable to get image dimensions.")

        # Get Image Date and Time
        self._date_time_fmt = "%Y:%m:%d %H:%M:%S"
        if "Image DateTime" in self._exif_data.keys():
            date_str = self._exif_data["Image DateTime"].printable
        elif "EXIF DateTimeOriginal" in self._exif_data.keys():
            date_str = self._exif_data["EXIF DateTimeOriginal"].printable
        else:
            logger.error("Date not available in exif.")
            return
        self._date_time = datetime.strptime(date_str, self._date_time_fmt)

    def extract_patch(self, limits: List[int]) -> np.ndarray:
        """Extract image patch
        Parameters
        __________
        - limits (List[int]): List containing the bounding box coordinates as: [xmin, ymin, xmax, ymax]
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

        Note:
            References for this functions can be found:

            * https://github.com/colmap/colmap/blob/e3948b2098b73ae080b97901c3a1f9065b976a45/src/util/bitmap.cc#L282
            * https://openmvg.readthedocs.io/en/latest/software/SfM/SfMInit_ImageListing/
            * https://photo.stackexchange.com/questions/40865/how-can-i-get-the-image-sensor-dimensions-in-mm-to-get-circle-of-confusion-from # noqa: E501

        Returns:
            K (np.ndarray): intrinsics matrix (3x3 numpy array).
        """
        if self._exif_data is None or len(self._exif_data) == 0:
            try:
                self.read_exif()
            except OSError:
                logger.error("Unable to read exif data.")
                return None
        try:
            focal_length_mm = float(self._exif_data["EXIF FocalLength"].printable)
        except OSError:
            logger.error("Focal length non found in exif data.")
            return None
        try:
            sensor_width_db = SensorWidthDatabase()
            sensor_width_mm = sensor_width_db.lookup(
                self._exif_data["Image Make"].printable,
                self._exif_data["Image Model"].printable,
            )
        except OSError:
            logger.error("Unable to get sensor size in mm from sensor database")
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

    def undistort_image(self, camera: Camera, out_path: str = None) -> np.ndarray:
        """
        undistort_image Wrapper around undistort_image function icepy4d.sfm.geometry module

        Args:
            camera (Camera): Camera object containing K and dist arrays.
            out_path (str, optional): Path for writing the undistorted image to disk. If out_path is None, undistorted image is not saved to disk. Defaults to None.

        Returns:
            np.ndarray: undistored image
        """
        self.read_image()

        und_imge = cv2.undistort(
            cv2.cvtColor(self._value_array, cv2.COLOR_RGB2BGR),
            camera.K,
            camera.dist,
            None,
            camera.K,
        )
        if out_path is not None:
            cv2.imwrite(out_path, und_imge)

        return und_imge


class ImageDS:
    """
    Class to manage Image datasets for multi epoch
    """

    def __init__(
        self,
        path: Union[str, Path, List[Path]],
        ext: str = None,
        recursive: bool = False,
    ) -> None:
        """
        Initializes an ImageDS object.

        Args:
            path (Union[str, Path]): Path to the image folder or list of paths of the images.
            ext (str, optional): Image extension for filtering files. If None is provided, all files in 'folder' are read. Defaults to None.
            recursive (bool, optional): Read files recursively. Defaults to False.

        Raises:
            IsADirectoryError: If the input path is invalid.
        """
        self._files = None
        self._timestamps = {}
        self._folder = None
        self._ext = None
        self._elem = 0

        if isinstance(path, list):
            assert all(
                [isinstance(x, Path) for x in path]
            ), "If a list is provided, all elements must be of type Path"
            self._files = path
        else:
            self._folder = Path(path)
            if not self._folder.exists():
                msg = f"Error: invalid input path {self._folder}"
                logger.error(msg)
                raise IsADirectoryError(msg)
            if ext is not None:
                self._ext = ext
            self._recursive = recursive
            self._read_image_list(self._folder)
        try:
            self._read_dates()
        except RuntimeError as err:
            logger.exception(err)

    def __len__(self) -> int:
        """
        Returns the number of images in the datastore.
        """
        return len(self._files)

    def __contains__(self, name: str) -> bool:
        """
        Checks if an image is in the datastore, given the image name.

        Args:
            name (str): The name of the image.

        Returns:
            bool: True if the image is in the datastore, False otherwise.
        """
        files = [x.name for x in self._files]
        return name in files

    def __getitem__(self, idx: int) -> str:
        """
        Returns the image path (including extension) at position idx in the datastore .

        Args:
            idx (int): The index of the image.

        Returns:
            str: The name of the image.
        """
        return str(self._files[idx])

    def __iter__(self):
        """
        Initializes the iterator for iterating over the images in the datastore.

        Returns:
            ImageDS: The iterator object.
        """
        self._elem = 0
        return self

    def __next__(self):
        """
        Returns the next image file in the iteration.

        Returns:
            Path: The next image file.

        Raises:
            StopIteration: If there are no more images in the datastore.
        """
        while self._elem < len(self):
            file = self._files[self._elem]
            self._elem += 1
            return file
        else:
            self._elem
            raise StopIteration

    def __repr__(self) -> str:
        """
        Returns a string representation of the datastore.

        Returns:
            str: The string representation of the datastore.
        """
        return f"ImageDS({self._folder}) with {len(self)} images."

    def reset_imageds(self) -> None:
        """
        Re-initializes the image datastore.
        """
        self._files = None
        self._folder = None
        self._ext = None
        self._elem = 0

    @property
    def files(self) -> List[Path]:
        """
        Returns the list of files in the datastore.
        """
        return self._files

    @property
    def folder(self) -> Path:
        """
        Returns the folder path of the datastore.
        """
        return self._folder

    @property
    def timestamps(self) -> Dict[int, datetime]:
        """
        Returns the timestamps of the images in the datastore.
        """
        return self._timestamps

    def _read_image_list(self, recursive: bool = None) -> None:
        """
        Reads the list of image files in the datastore.

        Args:
            recursive (bool, optional): Read files recursively. Defaults to None.

        Raises:
            AssertionError: If the image directory is invalid.
        """
        assert self._folder.is_dir(), "Error: invalid image directory."

        if recursive is not None:
            self._recursive = recursive
        if self._recursive:
            rec_patt = "**/"
        else:
            rec_patt = ""
        if self._ext is not None:
            ext_patt = f".{self._ext}"
        else:
            ext_patt = ""
        pattern = f"{rec_patt}*{ext_patt}"

        self._files = sorted(self._folder.glob(pattern))

        if len(self._files) == 0:
            logger.error(f"No images found in folder {self._folder}")
            return

    def _read_dates(self) -> None:
        """
        Reads the date and time for all the images in the ImageDS from the EXIF data.
        """
        assert self._files, "No image in ImageDS. Please read image list first"
        self._dates, self._times = {}, {}
        try:
            for id, im in enumerate(self._files):
                image = Image(im)
                self._timestamps[id] = image.datetime

                # These are kept only for backward compatibility
                self._dates[id] = image.date
                self._times[id] = image.time
        except:
            logger.error("Unable to read image dates and time from EXIF.")
            self._dates, self._times = {}, {}
            return

    def read_image(self, idx: int) -> Image:
        """
        Returns the image at the specified position as an Image instance, containing both EXIF and value data.

        Args:
            idx (int): The index of the image.

        Returns:
            Image: The Image instance.
        """
        image = Image(self._files[idx])
        image.read_image()
        return image

    def get_image_path(self, idx: int) -> Path:
        """
        Returns the path of the image at the specified position in the datastore.

        Args:
            idx (int): The index of the image.

        Returns:
            Path: The path of the image.
        """
        return self._files[idx]

    def get_image_stem(self, idx: int) -> str:
        """
        Returns the name without extension (stem) of the image at the specified position in the datastore.

        Args:
            idx (int): The index of the image.

        Returns:
            str: The name without extension of the image.
        """
        return self._files[idx].stem

    def get_image_timestamp(self, idx: int) -> datetime:
        """
        Returns the timestamp of the image at the specified position in the datastore.

        Args:
            idx (int): The index of the image.

        Returns:
            datetime: The timestamp of the image.
        """
        return self._timestamps[idx]

    def get_image_date(self, idx: int) -> str:
        """
        Returns the date of the image at the specified position in the datastore.

        Args:
            idx (int): The index of the image.

        Returns:
            str: The date of the image.
        """
        return self._dates[idx]

    def get_image_time(self, idx: int) -> str:
        """Return name without extension(stem) of the image at position idx in datastore"""
        return self._times[idx]

    def write_exif_to_csv(
        self, filename: str, sep: str = ",", header: bool = True
    ) -> None:
        assert self._folder.is_dir(), "Empty Image Datastore."
        file = open(filename, "w")
        if header:
            file.write("epoch,name,date,time\n")
        for i, img_path in enumerate(self._files):
            img = Image(img_path)
            name = img_path.name
            date = img.date
            time = img.time
            file.write(f"{i}{sep}{name}{sep}{date}{sep}{time}\n")
        file.close()


if __name__ == "__main__":
    """Test classes"""

    images = ImageDS("assets/img/cam1")

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

    # Undistort image
    # print(isinstance(image.undistort_image))

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
