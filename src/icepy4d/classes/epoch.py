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
from pathlib import Path
from typing import Dict, Union
from datetime import datetime as dt

import icepy4d.classes as classes


def parse_str_to_datetime(
    datetime: Union[str, dt], datetime_format: str = "%Y-%m-%d %H:%M:%S"
):
    if isinstance(datetime, dt):
        return datetime
    elif isinstance(datetime, str):
        try:
            datetime = dt.strptime(datetime, datetime_format)
        except:
            err = "Unable to convert datetime to string. You should provide a datetime object or a string in the format %Y-%m-%d %H:%M:%S, or you should pass the datetime format as a string to the datetime_format argument"
            logging.warning(err)
            raise ValueError(err)
    else:
        err = "Invalid epoch datetime. It should be a datetime object or a string in the format %Y-%m-%d %H:%M:%S"
    return datetime


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
        datetime: Union[str, dt],
        cameras: classes.Camera = None,
        images: classes.ImageDS = None,
        features: classes.Features = None,
        points: classes.Points = None,
        point_cloud: classes.PointCloud = None,
        epoch_dir: Union[str, Path] = None,
        datetime_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        """
        Initializes a Epcoh object with the provided data

        Args:
            cameras (classes.Camera): The dictionary of camera parameters
            images (classes.ImagesDict): The dictionary of images and their metadata
            features (classes.Features): The dictionary of feature points
            points (classes.Points): The dictionary of 3D points
        """
        self._datetime = parse_str_to_datetime(datetime, datetime_format)
        self.cameras = cameras
        self.images = images
        self.features = features
        self.points = points
        self.point_cloud = point_cloud
        self._epoch_dir = Path(epoch_dir) if epoch_dir else None

    @property
    def datetime(self):
        return self._datetime

    @property
    def epoch_dir(self):
        return self._epoch_dir

    def __repr__(self):
        """
        Returns a string representation of the Solution object

        Returns:
            str: The string representation of the Solution object
        """
        return f"Epoch {self.datetime}"

    def __iter__(self):
        """
        Returns an iterator over the four dictionaries of Solution object

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
        Computes the hash value of the Solution object

        Returns:
            int: The hash value of the Solution object
        """
        return hash((self.cameras, self.images, self.features, self.points))

    def save_pickle(self, path: Union[str, Path]) -> bool:
        """
        Saves the Solution object to a binary file

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
            logging.error("Unable to save the Solution as Pickle object")
            return False

    @staticmethod
    def read_pickle(path: Union[str, Path], ignore_errors: bool = False):
        """
        Loads a Solution object from a binary file

        Args:
            path (Union[str, Path]): The path to the binary file

        Returns:
            Solution: A Solution object
        """
        path = Path(path)
        if not ignore_errors:
            assert path.exists(), "Input path does not exists"

        try:
            with open(path, "rb") as inp:
                solution = pickle.load(inp)
            return solution
        except:
            logging.error("Unable to read Solution from pickle file")
            return None


class Epoches:
    """Class for storing all the epochs in ICEpy4D processing"""

    def __init__(self) -> None:
        self._last_epoch: int = -1
        self._epochs: Dict[int, Epoch] = {}
        self._epoches_map: Dict[int, dt] = {}

    def __repr__(self):
        """
        Returns a string representation of the Epoches object

        Returns:
            str: The string representation of the Epoches object
        """

        return f"Epoches with {len(self._epochs)} epochs"

    def __iter__(self):
        """
        Returns an iterator over the four dictionaries of Solution object

        Yields:
            dict: The dictionary of camera parameters
            dict: The dictionary of images and their metadata
        """
        yield self.epochs

    def __hash__(self):
        """
        Computes the hash value of the Epoches object

        Returns:
            int: The hash value of the Epoches object
        """
        return hash((self._epochs))

    def __getitem__(self, epoch_id):
        """
        Returns the epoch object with the provided epoch_id

        Args:
            epoch_id (int): The numeric key of the epoch

        Returns:
            Epoch: The epoch object
        """
        return self._epochs[epoch_id]

    def add_epoch(self, epoch: Epoch):
        """
        Adds an epoch to the Epoches object

        Args:
            epoch_id (int): The numeric key of the epoch
            epoch_date (str): The corresponding date of the epoch
            epoch (Epoch): The epoch object to be added
        """
        assert isinstance(epoch, Epoch), "Input epoch must be of type Epoch"
        assert hasattr(epoch, "datetime"), "Epoch must have a datetime attribute"
        assert isinstance(epoch.datetime, dt), "Epoch datetime must be of type datetime"
        assert (
            epoch.datetime not in self._epoches_map.values()
        ), "Epoch with the same date already exists"

        epoch_id = self._last_epoch + 1
        self._epoches_map[epoch_id] = epoch.datetime
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
        return self._epochs.get(epoch_id).datetime

    def get_epoch_id(
        self, epoch_date: Union[str, dt], datetime_format: str = "%Y-%m-%d %H:%M:%S"
    ) -> int:
        datetime = parse_str_to_datetime(epoch_date, datetime_format)
        for i, ep in self._epochs.items():
            if ep.datetime == datetime:
                return i

    def get_epoch_by_date(
        self, datetime: Union[str, dt], datetime_format: str = "%Y-%m-%d %H:%M:%S"
    ) -> Epoch:
        datetime = parse_str_to_datetime(datetime, datetime_format)
        for ep in self._epochs.values():
            if ep.datetime == datetime:
                return ep
        logging.warning(f"Epoch with datetime {datetime} not found")
        return None


if __name__ == "__main__":
    # Epoch from datetime object
    date = dt.strptime("2021-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    ep = Epoch(datetime=date)
    print(ep)

    # Epoch from string
    date = "2021-01-01 00:00:00"
    ep = Epoch(datetime=date)
    print(ep)

    # Epoches
    epoches = Epoches()
    print(epoches)

    # Add epoch
    ep1 = Epoch(datetime=date)
    epoches.add_epoch(ep)
    print(epoches)

    # Get epoch
    print(epoches.get_epoch_by_date(date))
    print(epoches.get_epoch_by_date("2021-01-02 00:00:00"))

    print(epoches.get_epoch_date(0))
    print(epoches.get_epoch_id(date))

    ep2 = Epoch(datetime="2021-01-02 00:00:00")
    epoches.add_epoch(ep2)
    print(epoches)

    print(epoches.get_epoch_date(1))
    print(epoches[1])

    print("Done")
