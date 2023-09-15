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
import pickle
from datetime import datetime as dt
from datetime import timedelta
from pathlib import Path
from copy import deepcopy
from typing import Dict, Union, List, Tuple
import os

from .camera import Camera
from .features import Features
from .point_cloud import PointCloud
from .images import Image, ImageDS
from .targets import Targets
from .points import Points

logger = logging.getLogger(__name__)


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_str_to_datetime(
    datetime: Union[str, dt], datetime_format: str = DEFAULT_DATETIME_FMT
):
    """
    Parse a string or datetime object into a datetime object.

    Args:
        datetime (Union[str, dt]): A string or datetime object to be parsed.
        datetime_format (str): A format string specifying the datetime format
            if `datetime` is a string. Default format is "%Y-%m-%d_%H:%M:%S".

    Returns:
        dt: A datetime object representing the parsed datetime.

    Raises:
        ValueError: If the input cannot be converted to a datetime object.
    """
    if isinstance(datetime, dt):
        return datetime
    elif isinstance(datetime, str):
        try:
            datetime = dt.strptime(datetime, datetime_format)
        except:
            err = f"Unable to convert datetime to string. You should provide a datetime object or a string in the format {DEFAULT_DATETIME_FMT}, or you should pass the datetime format as a string to the datetime_format argument"
            logger.warning(err)
            raise ValueError(err)
    else:
        err = f"Invalid epoch datetime. It should be a datetime object or a string in the format {DEFAULT_DATETIME_FMT}"
    return datetime


def find_closest_timestamp(
    ref_timestamp: dt,
    timestamps: List[dt],
    time_tolerance: timedelta = timedelta(seconds=60),
) -> Tuple[dt, int, timedelta]:
    """
    Find the closest timestamp to a reference timestamp within a list of timestamps.

    Args:
        ref_timestamp (dt): The reference datetime object.
        timestamps (List[dt]): A list of datetime objects to search for the closest timestamp.
        time_tolerance (timedelta): The maximum time difference allowed to consider a timestamp as close.
            Default is 60 seconds.

    Returns:
        Tuple[dt, int, timedelta]: A tuple containing the closest timestamp, its index in the list of timestamps,
        and the time difference between the reference timestamp and the closest timestamp.
        If no timestamp within the specified tolerance is found, the timestamp is None, the index is None,
        and the time difference is None.
    """
    tdiffs = [abs(ts - ref_timestamp) for ts in timestamps]
    min_dt = min(tdiffs) if min(tdiffs) < time_tolerance else None
    if min_dt is not None:
        closest_idx = tdiffs.index(min_dt)
        closest_timestamp = timestamps[closest_idx]
    else:
        closest_idx = None
        closest_timestamp = None
    return closest_timestamp, closest_idx, min_dt


class EpochDataMap(dict):
    """
    A class for managing epoch data mapping, including timestamps and associated images.

    Args:
        image_dir (Union[str, Path]): The directory containing image data.
        master_camera (str): The name of the master camera (optional).
        time_tolerance_sec (int): The maximum time difference allowed to consider two images taken by different cameras as simultaneous (this allows for considering a non-perfect time synchronization between different cameras). Default is 1200 seconds (20 minutes).

    Attributes:
        _image_dir (Path): The path to the image directory.
        _master_camera (str): The name of the master camera.
        _timetolerance (timedelta): The time tolerance for timestamp matching.
        _cams (List[str]): The list of camera names.
        _map (dict): The mapping of epoch data.

    Methods:
        _get_timestamps(folder: Union[str, Path]) -> Tuple[List[dt], List[Path]]:
            Get timestamps and image paths from a specified folder.
        _build_map():
            Build the mapping of epoch data.
        _write_map(filename: str, sep: str = ",", header: bool = True) -> None:
            Write the mapping data to a CSV file.
        __getitem__(self, key):
            Retrieve epoch data by key.
        __repr__(self) -> str:
            Return a string representation of the EpochDataMap.
        __len__(self) -> int:
            Return the number of epochs in the EpochDataMap.
        __iter__(self):
            Initialize an iterator for the EpochDataMap.
        __next__(self):
            Get the next element in the EpochDataMap.
        __contains__(self, timestamp: Union[str, dt]) -> bool:
            Check if a timestamp is present in the EpochDataMap.
        get_epoch_timestamp(self, epoch_id: int) -> dt:
            Get the timestamp of a specific epoch.
        get_epoch_images(self, epoch_id: int) -> List[Path]:
            Get the images associated with a specific epoch.
        get_epoch_images_by_timestamp(self, timestamp: Union[str, dt]) -> List[Path]:
            Get the images associated with an epoch by timestamp.
        get_epoch_image_timestamps(self, epoch_id: int) -> List[dt]:
            Get the timestamps of images associated with a specific epoch.
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        master_camera=None,
        time_tolerance_sec: timedelta = 1200,
    ):
        """
        Initialize the EpochDataMap with image directory, master camera, and time tolerance.
        """
        self._image_dir = Path(image_dir)
        assert self._image_dir.exists(), f"{self._image_dir} does not exist"

        # if a master camera is specified, check if it exists in the image dir
        if master_camera:
            assert self._master_camera in os.scandir(
                self._image_dir
            ), f"Master camera not found in image directory {self._image_dir}"
        # if master camera is not specified, use the first camera found
        else:
            master_camera = sorted(
                [f.name for f in os.scandir(self._image_dir) if f.is_dir()]
            )[0]
        self._master_camera = master_camera

        self._timetolerance = timedelta(seconds=time_tolerance_sec)

        # Get camera names
        self._cams = sorted([f.name for f in os.scandir(self._image_dir) if f.is_dir()])

        # Build dict
        self._map = {}
        self._build_map()

        # Write dict to file
        self._write_map(self._image_dir / "epoch_map.csv")

    def __getitem__(self, key):
        return self._map[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with {len(self._map)} epochs"

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        self._elem = 0
        return self

    def __next__(self):
        while self._elem < len(self._map):
            file = self._map[self._elem]
            self._elem += 1
            return file
        else:
            self._elem = 0
            raise StopIteration

    def __contains__(self, timestamp: Union[str, dt]) -> bool:
        timestamp = parse_str_to_datetime(timestamp)
        timestamps = [x["timestamp"] for x in self._map.values()]
        return timestamp in timestamps

    @property
    def cams(self):
        return self._cams

    def get_epoch_timestamp(self, epoch_id: int) -> dt:
        return str(self._map[epoch_id]["timestamp"]).replace(" ", "_")

    def get_epoch_images(self, epoch_id: int) -> List[Path]:
        return self._map[epoch_id]["images"]

    def get_epoch_images_by_timestamp(self, timestamp: Union[str, dt]) -> List[Path]:
        timestamp = parse_str_to_datetime(timestamp)
        timestamps = [x["timestamp"] for x in self._map.values()]
        idx = timestamps.index(timestamp)
        return self._map[idx]["images"]

    def get_epoch_image_timestamps(self, epoch_id: int) -> List[dt]:
        return self._map[epoch_id]["image_timestamps"]

    def _get_timestamps(self, folder: Union[str, Path]) -> Tuple[List[dt], List[Path]]:
        imageDS = ImageDS(folder)
        paths = list(imageDS.files)
        timestamps = list(imageDS.timestamps.values())
        return timestamps, paths

    def _build_map(self):
        # Build imageDS for master camera and get timestamps
        timestamps, paths = self._get_timestamps(self._image_dir / self._master_camera)

        # build mapping dict for master camera
        for i, (ts, path) in enumerate(zip(timestamps, paths)):
            self._map[i] = AttributeDict(
                {
                    "timestamp": ts,
                    "images": {self._master_camera: Image(path)},
                }
            )

        # Find closest timestamp for each camera
        slave_cameras = deepcopy(self._cams)
        slave_cameras.remove(self._master_camera)

        for cam in slave_cameras:
            timestamps1, paths1 = self._get_timestamps(self._image_dir / cam)
            for key, value in self._map.items():
                ref_ts = value["timestamp"]
                _, closest_idx, _ = find_closest_timestamp(
                    ref_ts, timestamps1, self._timetolerance
                )
                self._map[key]["images"][cam] = Image(paths1[closest_idx])

    def _write_map(self, filename: str, sep: str = ",", header: bool = True) -> None:
        file = open(filename, "w")
        if header:
            columns = ["epoch", "date", "time"]
            for cam in self._cams:
                columns.append(cam)
                columns.append(f"{cam}_timestamp")
            file.write(f"{sep}".join(columns) + "\n")
        for key, value in self._map.items():
            value = self._map[key]
            date = value["timestamp"].strftime("%Y-%m-%d")
            time = value["timestamp"].strftime("%H:%M:%S")
            str_2_add = []
            for cam in self._cams:
                str_2_add.append(value["images"][cam].name)
                str_2_add.append(
                    f"{value['images'][cam].date}_{value['images'][cam].time}"
                )
            line = [str(key), date, time] + str_2_add
            file.write(f"{sep}".join(line) + "\n")
        file.close()


class Epoch:
    """
    Class for storing, saving, and reading ICEpy4D solution at one epoch

    Attributes:
        cameras (classes.Camera): The dictionary of camera parameters
        images (classes.ImagesDict): The dictionary of images and their metadata
        features (classes.Features): The dictionary of feature points
        points (classes.Points): The dictionary of 3D points
    """

    def __init__(
        self,
        timestamp: Union[str, dt],
        epoch_dir: Union[str, Path] = None,
        images: Dict[str, Image] = None,
        cameras: Dict[str, Camera] = None,
        features: Dict[str, Features] = None,
        points: Points = None,
        targets: Targets = None,
        point_cloud: PointCloud = None,
        datetime_format: str = DEFAULT_DATETIME_FMT,
    ) -> None:
        """
        Initializes a Epcoh object with the provided data

        Args:
            cameras (classes.Camera): The dictionary of camera parameters
            images (classes.ImagesDict): The dictionary of images and their metadata
            features (classes.Features): The dictionary of feature points
            points (classes.Points): The dictionary of 3D points

        TODO: Simplify the initialization of the Epoch object if no parameters are provided.
        """

        self._timestamp = parse_str_to_datetime(timestamp, datetime_format)
        self.images = images
        self.cameras = cameras
        self.features = features
        self.points = points
        self.targets = targets
        self.point_cloud = point_cloud

        if epoch_dir is not None:
            self.epoch_dir = Path(epoch_dir)
        else:
            logger.info("Epoch directory not provided. Using epoch timestamp.")
            self.epoch_dir = Path(str(self._timestamp).replace(" ", "_"))
        self.epoch_dir.mkdir(parents=True, exist_ok=True)

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def date_str(self) -> str:
        """
        Returns the date and time of the epoch in a string.

        Returns:
            str: The date and time of the epoch in the format "YYYY:MM:DD HH:MM:SS".
        """
        return self._timestamp.strftime("%Y:%m:%d")

    @property
    def time_str(self) -> str:
        """
        Returns the time of the epoch as a string.

        Returns:
            str: The time of the epoch in the format "HH:MM:SS".

        """
        return self._timestamp.strftime("%H:%M:%S")

    def __str__(self) -> str:
        """
        Returns a string representation of the Epoch object

        Returns:
            str: The string representation of the Epoch object
        """
        return f"{self._timestamp.strftime(DEFAULT_DATETIME_FMT).replace(' ', '_')}"

    def __repr__(self):
        """
        Returns a string representation of the Epoch object

        Returns:
            str: The string representation of the Epoch object
        """
        return f"Epoch {self._timestamp}"

    def __iter__(self):
        """
        Returns an iterator over the four dictionaries of Epoch object

        Yields:
            dict: The dictionary of camera parameters
            dict: The dictionary of images and their metadata
            dict: The dictionary of feature points
            dict: The dictionary of 3D points
        """
        yield self.cameras
        yield self.images
        yield self.features
        yield self.points

    def __hash__(self):
        """
        Computes the hash value of the Epoch object

        Returns:
            int: The hash value of the Epoch object
        """
        return hash((self.cameras, self.images, self.features, self.points))

    def save_pickle(self, path: Union[str, Path]) -> bool:
        """
        Saves the Epoch object to a binary file

        Args:
            path (Union[str, Path]): The path to the binary file

        Returns:
            bool: True if the object was successfully saved to file, False otherwise
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            return True
        except:
            logger.error("Unable to save the Solution as Pickle object")
            return False

    @staticmethod
    def read_pickle(path: Union[str, Path]):
        """
        Load a Epoch object from a binary file

        Args:
            path (Union[str, Path]): The path to the binary file

        Returns:
            Epoch: An Epoh object
        """
        path = Path(path)
        assert path.exists(), f"Input path {path} does not exists"

        try:
            logger.info(f"Loading epoch from {path}")
            with open(path, "rb") as inp:
                epoch = pickle.load(inp)
            if epoch is not None:
                logger.info(f"Epoch loaded from {path}")
                return epoch
            else:
                logger.error(f"Unable to load epoch from {path}")
                return None
        except Exception as e:
            raise e(f"Unable to read Epoch from file {path}")


class Epoches:
    """Class for storing all the epochs in ICEpy4D processing"""

    def __init__(self, starting_epoch: int = 0) -> None:
        self._starting_epoch = starting_epoch
        self._last_epoch: int = -1
        self._epochs: Dict[int, Epoch] = {}
        self._epoches_map: Dict[int, dt] = {}
        self._elem = 0

    def __repr__(self):
        """
        Returns a string representation of the Epoches object

        Returns:
            str: The string representation of the Epoches object
        """

        return f"Epoches with {len(self._epochs)} epochs"

    def __len__(self) -> int:
        """Get number of epoches in the Epoches object"""
        return len(self._epochs)

    def __iter__(self):
        self._elem = self._starting_epoch
        return self

    def __next__(self):
        while self._elem <= self._last_epoch:
            file = self._epochs[self._elem]
            self._elem += 1
            return file
        else:
            self._elem
            raise StopIteration

    def __getitem__(self, epoch_id):
        """
        Returns the epoch object with the provided epoch_id

        Args:
            epoch_id (int): The numeric key of the epoch

        Returns:
            Epoch: The epoch object
        """
        return self._epochs[epoch_id]

    def __contains__(
        self, epoch_date: Union[str, dt], datetime_format: str = DEFAULT_DATETIME_FMT
    ) -> bool:
        """Check if an epoch is in the Epoch objet"""
        timestamp = parse_str_to_datetime(epoch_date, datetime_format)
        timestamps = [x for x in self._epoches_map.values()]
        return timestamp in timestamps

    def add_epoch(self, epoch: Epoch):
        """
        Adds an epoch to the Epoches object

        Args:
            epoch_id (int): The numeric key of the epoch
            epoch_date (str): The corresponding date of the epoch
            epoch (Epoch): The epoch object to be added
        """
        assert isinstance(epoch, Epoch), "Input epoch must be of type Epoch"
        assert hasattr(epoch, "timestamp"), "Epoch must have a timestamp attribute"
        assert isinstance(
            epoch.timestamp, dt
        ), "Epoch timestamp must be of type datetime"
        assert (
            epoch.timestamp not in self._epoches_map.values()
        ), "Epoch with the same timestamp already exists"
        if self._last_epoch == -1:
            epoch_id = self._starting_epoch
        else:
            epoch_id = self._last_epoch + 1
        self._epoches_map[epoch_id] = epoch.timestamp
        self._epochs[epoch_id] = epoch
        self._last_epoch = epoch_id

    def get_epoch_date(self, epoch_id: int) -> str:
        """
        Retrieves the date corresponding to an epoch from the Epoches object

        Args:
            epoch_id (int): The numeric key of the epoch

        Returns:
            str: The date corresponding to the epoch_id
        """
        return self._epochs.get(epoch_id).timestamp

    def get_epoch_id(
        self, epoch_date: Union[str, dt], datetime_format: str = DEFAULT_DATETIME_FMT
    ) -> int:
        timestamp = parse_str_to_datetime(epoch_date, datetime_format)
        for i, ep in self._epochs.items():
            if ep.timestamp == timestamp:
                return i

    def get_epoch_by_date(
        self, timestamp: Union[str, dt], datetime_format: str = DEFAULT_DATETIME_FMT
    ) -> Epoch:
        timestamp = parse_str_to_datetime(timestamp, datetime_format)
        for ep in self._epochs.values():
            if ep.timestamp == timestamp:
                return ep
        logger.warning(f"Epoch with timestamp {timestamp} not found")
        return None


if __name__ == "__main__":
    epoch_map = EpochDataMap("/home/francesco/Projects/icepy4d/data/img")

    # Epoch from datetime object
    date = dt.strptime("2021-01-01 00:00:00", DEFAULT_DATETIME_FMT)
    ep = Epoch(timestamp=date)
    print(ep)

    # Epoch from string
    date = "2021-01-01 00:00:00"
    ep = Epoch(timestamp=date)
    print(ep)

    # Epoches
    epoches = Epoches()
    print(epoches)

    # Add epoch
    ep1 = Epoch(timestamp=date)
    epoches.add_epoch(ep)
    print(epoches)

    # Get epoch
    print(epoches.get_epoch_by_date(date))
    print(epoches.get_epoch_by_date("2021-01-02 00:00:00"))

    print(epoches.get_epoch_date(0))
    print(epoches.get_epoch_id(date))

    ep2 = Epoch(timestamp="2021-01-02 00:00:00")
    epoches.add_epoch(ep2)
    print(epoches)

    print(epoches.get_epoch_date(1))
    print(epoches[1])

    for ep in epoches:
        print(ep)

    # Check if epoch is in epoches
    print("2021-01-01 00:00:00" in epoches)

    print("Done")
