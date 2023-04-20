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

import pickle
import numpy as np
import logging
import time

from typing import Union, List, Tuple
from pathlib import Path

from typing import List, Union

if __name__ == "__main__":
    from src.icepy4d.classes.camera import Camera
    from src.icepy4d.classes.point_cloud import PointCloud
else:
    from .camera import Camera
    from .point_cloud import PointCloud


def float32_type_check(
    array: np.ndarray, cast_integers: bool = False, verbose: bool = False
) -> np.ndarray:
    """
    float32_type_check Check if the numpy array is of type np.float32 and, if possible, return a casted array to np.float32

    Args:
        array (np.ndarray): Numpy array
        cast_integers (bool, optional): Cast integers to float. Defaults to False.
        verbose (bool, optional): Log output. Defaults to False.

    Raises:
        ValueError: Raise ValueError if the function is unable to safely cast the array to np.float32

    Returns:
        np.ndarray: np.float32 numpy array
    """
    if array.dtype == np.float64 or array.dtype == float:
        if verbose:
            logging.info("Input array are float64 numbers. Casting them to np.float32")
        array = array.astype(np.float32)

    if cast_integers:
        if array.dtype == int or array.dtype == np.int32 or array.dtype == np.int64:
            if verbose:
                logging.info("Input array are int numbers. Casting them to np.float32")
            array = array.astype(np.float32)
    if array.dtype != np.float32:
        raise ValueError(
            "Invalid type of input array. It must be a numpy array of type np.float32"
        )

    return array


class Point:
    """
    Point Class for storing 3D point information and relative modules for getting image projections and building point clouds

    Attributes:
        _track_id (int): Univocal track ID identifying the point and the corresponding features on the images.
        _X (float): X coordinate of the point.
        _Y (float): Y coordinate of the point.
        _Z (float): Z coordinate of the point.
        _color (np.ndarray): (3,) or (3,1) or (1,3) numpy array of type np.float32 with the point colors, as float values in the range [0,1].
        _cov (np.ndarray): 3x3 numpy array containing the covariance matrix.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        track_id: int = None,
        color: np.ndarray = None,
        cov: np.ndarray = None,
    ) -> None:
        """
        __init__ Create Point object

        Args:
            coordinates (np.ndarray): (3,) or (3,1) or (1,3) numpy array of type np.float32 containing X Y Z coordinates of the point.
            track_id (np.int32): univocal track_id identifying the point and the corresponding featues on the images. Defaults to None.. Defaults to None.
            color (np.ndarray, optional): (3,) or (3,1) or (1,3) numpy array of type np.float32 with the point colors, as float values in the range [0,1]. Defaults to None.
            cov (np.ndarray, optional): 3x3 numpy array containing the covariance matrix. Defaults to None.
        """

        assert isinstance(coordinates, np.ndarray), "invalid argument coordinates"
        if coordinates.shape == (3, 1) or coordinates.shape == (3, 1):
            coordinates = coordinates.squeeze()
        assert coordinates.shape == (
            3,
        ), "Invalid shape of coordinates array. It must be a (3,) numpy array (vector)"
        coordinates = float32_type_check(coordinates, cast_integers=True)

        assert isinstance(color, np.ndarray), "invalid argument color"
        if color.shape == (3, 1) or color.shape == (3, 1):
            color = color.squeeze()
        assert color.shape == (
            3,
        ), "Invalid shape of color array. It must be a (3,) numpy array"
        color = float32_type_check(color, cast_integers=False)
        assert (
            color.dtype == np.float32
        ), "Invalid type of color vector. It must be of type np.float32"

        self._track_id = track_id
        self._X = coordinates[0]
        self._Y = coordinates[1]
        self._Z = coordinates[2]
        self._color = color
        self._cov = cov

    # Setters

    # Getters
    @property
    def track_id(self):
        return self._track_id

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def Z(self):
        return self._Z

    @property
    def coordinates(self):
        return np.array([self._X, self._Y, self._Z], dtype=np.float32)

    @property
    def color(self):
        return self._color.astype(np.float32)

    def project(self, camera: Camera) -> np.ndarray:
        """
        project project the 3D point to the camera and return image coordinates

        Args:
            camera (Camera): Camera object containing extrinsics and intrinsics

        Returns:
            np.ndarray: coordinates of the projection in px
        """
        return camera.project_point(self.coordinates.reshape(1, 3))


class Points:
    """
    Points class represents a collection of several 3D Point objects.

    Attributes:
        _values (dict): A dictionary that maps Point IDs to Point objects.
        _last_id (int): The last assigned Point ID.
        _iter (int): An iterator used to iterate over the point IDs.

    Note:
        Use getters and setters methods to access to the features stored in Points object.
    """

    def __init__(self):
        self._values = {}
        self._last_id = -1
        self._iter = 0
        """
        __init__ Initialize Points object.
        """

    def __len__(self) -> int:
        """
        __len__ Get number of points stored in Features object

        Returns:
            int: number of features
        """
        return len(self._values)

    def __getitem__(self, track_id: np.int32) -> Point:
        """
        __getitem__ Get Point object by calling Features instance with [] based on track_id (e.g., features[track_id] to get first Feature object)

        Args:
            track_id (int): track_id of the feature to extract

        Returns:
            Feature: requested Feature object
        """
        if track_id in list(self._values.keys()):
            return self._values[track_id]
        else:
            logging.warning(f"Feature with track id {track_id} not available.")
            return None

    def __contains__(self, track_id: np.int32) -> bool:
        """
        __contains__ Check if a feature with given track_id is present in Features object.

        Args:
            track_id (np.int32): track_id of the feature to check

        Returns:
            bool: True if the feature is present.
        """
        if track_id in list(self._values.keys()):
            return True
        else:
            return False

    def __delitem__(self, track_id: np.int32) -> bool:
        """
        __delitem__ Deleate a feature with its given track_id, if this is present in Features object (log a warning otherwise).

        Args:
            track_id (np.int32): track_id of the feature to delete

        Returns:
            bool: True if the item was present and deleted, False otherwise.
        """
        if track_id not in self:
            logging.warning(f"Feature with track_id {track_id} not present")
            return False
        else:
            del self._values[track_id]
            return True

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        while self._iter < len(self):
            f = self._values[self._iter]
            self._iter += 1
            return f
        else:
            self._iter = 0
            raise StopIteration

    @property
    def num_points(self):
        """
        num_features Number of points stored in Features object
        """
        return len(self._values)

    @property
    def last_track_id(self):
        """
        last_track_id Track_id of the last point stored in Features object
        """
        return self._last_id

    def get_track_ids(self) -> Tuple[np.int32]:
        """
        get_track_it Get a ordered tuple of track_id of all the points

        Returns:
            tuple: tuple of size (n,) with track_ids
        """
        return tuple([np.int32(x) for x in self._values.keys()])

    def append_point(self, new_point: Point) -> None:
        """
        append_point append a single Feature object to Features.

        Args:
            new_point (Point): Feature object to be appended.
        """
        assert isinstance(
            new_point, Point
        ), "Invalid input feature. It must be Point object"
        self._last_id += 1
        self._values[self._last_id] = new_point

    def set_last_track_id(self, last_track_id: np.int32) -> None:
        """
        set_last_track_id set track_id of last point to a custom value

        Args:
            last_track_id (np.int32): track_id to set.
        """
        try:
            last_id = np.int32(last_track_id)
        except:
            raise ValueError(
                "Invalid input argument last_track_id. It must be an integer number."
            )
        self._last_id = last_id

    def append_points_from_numpy(
        self,
        coordinates: np.ndarray,
        track_ids: List[np.int32] = None,
        colors: np.ndarray = None,
    ) -> None:
        """
        append_points_from_numpy append new features to Features object, starting from a nx3 numpy array containing XYZ coordinates.

        Args:
            coordinates (np.ndarray): nx3 numpy array containing x coordinates of all keypoints
            track_ids (List[int]): Sorted list containing the track_id of each point to be added to Points object. Default to None.
            colors (np.ndarray): nx3 numpy array containing colors as float number in range [0,1]
        """

        if not np.any(coordinates):
            logging.warning("Empty input feature arrays. Nothing done.")
            return None
        assert isinstance(coordinates, np.ndarray), "invalid argument coordinates"
        assert (
            coordinates.shape[1] == 3
        ), "Invalid shape of coordinates array. It must be a nx3 numpy array"
        coordinates = float32_type_check(coordinates, cast_integers=True)

        if track_ids is None:
            ids = range(self._last_id + 1, self._last_id + len(coordinates) + 1)
        else:
            assert isinstance(track_ids, list) or isinstance(
                track_ids, tuple
            ), "Invalid track_ids input. It must be a list or a tuple of integers of the same size of the input arrays."
            assert len(track_ids) == len(
                coordinates
            ), "invalid size of track_id input. It must be a list of the same size of the input arrays."

            try:
                for id in track_ids:
                    if id in list(self._values.keys()):
                        msg = f"Feature with track_id {id} is already present in Features object. Ignoring input track_id and assigning progressive track_ids."
                        logging.error(msg)
                        raise ValueError(msg)
                ids = track_ids
            except ValueError:
                ids = range(self._last_id + 1, self._last_id + len(coordinates) + 1)

        if colors is not None:
            colors = np.float32(colors)
        else:
            colors = [None for _ in range(len(coordinates))]

        for (id, coor, col) in zip(ids, coordinates, colors):
            self._values[id] = Point(coor, id, color=col)
            self._last_id = id

    def to_numpy(self) -> np.ndarray:
        """
        to_numpy Get all points' coordinates stacked as numpy array.

        Returns:
            np.ndarray: nx3 numpy array of type np.float32 with XYZ coordinates
        """
        pts = np.empty((len(self), 3), dtype=np.float32)
        for i, v in enumerate(self._values.values()):
            pts[i, :] = np.float32(v.coordinates)

        return pts

    def colors_to_numpy(self, as_uint8: bool = False) -> np.ndarray:
        """
        colors_to_numpy Get points' colors stacked as numpy array.


        Args:
            as_uint8 (bool, optional): Convert RGB colors to integers numbers at 8bit (np.uint8) with values ranging between 0 and 255. Defaults to False.

        Returns:
            np.ndarray: nx3 numpy array with RGB colors (either in as floating numbers or integers ranging between [0, 255])
        """
        if as_uint8:
            cols = np.empty((len(self), 3), dtype=np.uint8)
            for i, v in enumerate(self._values.values()):
                cols[i, :] = np.uint8(v.color * 255)
        else:
            cols = np.empty((len(self), 3), dtype=np.float32)
            for i, v in enumerate(self._values.values()):
                cols[i, :] = np.float32(v.color)

        return cols

    def to_point_cloud(self) -> PointCloud:
        """
        to_point_cloud Convert Points object to PointCloud object that store the data with Open3D class, has methods to visualize the point cloud and save it

        Returns:
            PointCloud: PointCloud object
        """

        pcd = PointCloud(points3d=self.to_numpy(), points_col=self.colors_to_numpy())
        return pcd

    def reset_points(self):
        """Reset Points instance"""
        self._values = {}
        self._last_id = -1
        self._iter = 0

    def filter_point_by_mask(
        self, inlier_mask: List[bool], verbose: bool = False
    ) -> None:
        """
        delete_feature_by_mask Keep only inlier features, given a mask array as a list of boolean values. Note that this function does NOT take into account the track_id of the features! Inlier mask must have the same lenght as the number of points stored in the Features instance.

        Args:
            inlier_mask (List[bool]): boolean mask with True value in correspondance of the points to keep. inlier_mask must have the same length as the total number of features.
            verbose (bool): log number of filtered features. Defaults to False.
        """
        assert np.array_equal(
            inlier_mask, inlier_mask.astype(bool)
        ), "Invalid type of input argument for inlier_mask. It must be a boolean vector with the same lenght as the number of points stored in the Points object."
        assert len(inlier_mask) == len(
            self
        ), "Invalid shape of input argument for inlier_mask. It must be a boolean vector with the same lenght as the number of points stored in the Points object."

        feat_idx = list(self._values.keys())
        indexes = [feat_idx[i] for i, x in enumerate(inlier_mask) if x]
        self.filter_points_by_index(indexes, verbose=verbose)

    def filter_points_by_index(
        self, indexes: List[np.int32], verbose: bool = False
    ) -> None:
        """
        delete_feature_by_mask Keep only inlier points, given a list of index (int values) of the points to keep.

        Args:
            indexes (List[int]): List with the index of the points to keep.
            verbose (bool): log number of filtered points. Defaults to False.

        """
        new_dict = {k: v for k, v in self._values.items() if v.track_id in indexes}
        if verbose:
            logging.info(
                f"Points filtered: {len(self)-len(new_dict)}/{len(self)} removed. New Points size: {len(new_dict)}."
            )
        last_id = list(new_dict.keys())[-1]
        self._values = new_dict
        self._last_id = last_id

    def get_points_by_index(self, indexes: List[np.int32]) -> dict:
        """
        get_feature_by_index Get inlier points, given a list of index (int values) of the points to keep.

        Args:
            indexes (List[int]): List with the index of the points to keep.

        Returns:
            dict: dictionary containing the selected points with track_id as keys and Point object as values {track_id: Point}
        """
        return {k: v for k, v in self._values.items() if v.track_id in indexes}

    def save_as_txt(
        self,
        path: Union[str, Path],
        fmt: str = "%i",
        delimiter: str = ",",
        header: str = "x,y",
    ):
        """Save keypoints in a .txt file"""
        pts = self.to_numpy()
        np.savetxt(path, pts, fmt=fmt, delimiter=delimiter, newline="\n", header=header)

    def save_as_pickle(self, path: Union[str, Path]) -> True:
        """Save keypoints in as pickle file"""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    """Test classes"""

    from src.icepy4d.utils import setup_logger

    setup_logger()

    # out = float32_type_check(2.4)

    n_feat = 10000
    coord = np.random.rand(n_feat, 3)
    cols = np.random.rand(n_feat, 3)

    points = Points()
    t0 = time.time()
    points.append_points_from_numpy(coordinates=coord, colors=cols)
    t1 = time.time()
    logging.info(
        f"Append features from numpy array to dict of Feature objects: Elapsed time {t1-t0:.4f} s"
    )

    logging.info("Convert points to point cloud")
    pcd = points.to_point_cloud()
    print(pcd.get_colors())
