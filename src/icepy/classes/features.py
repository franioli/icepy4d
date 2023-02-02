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
from copy import deepcopy

from .camera import Camera


class Feature:
    __slots__ = ("_x", "_y", "_track", "_descr", "_score", "epoch")

    def __init__(
        self,
        x: float,
        y: float,
        track_id: np.int64 = None,
        descr: np.ndarray = None,
        score: np.float32 = None,
        epoch: np.int64 = None,
    ) -> None:
        """
        __init__ Create Feature object

        Args:
            x (Union[float, int]): x coordinate (as for OpenCV image coordinate system)
            y (Union[float, int]): y coordinate (as for OpenCV image coordinate system)
            track_id (np.int64, optional): track_id. Defaults to None.
            descr (np.ndarray, optional): descriptor as a numpy array with 128 or 256 elements. Defaults to None.
            score (float, optional): score. Defaults to None.
            epoch (np.int64, optional): Epoch in which the feature is detected (or belongs to). Defaults to None.

        Raises:
            AssertionError: Invalid shape of the descriptor. It must be a numpy array with 128 or 256 elements.
        """
        self._x = np.float32(x)
        self._y = np.float32(y)

        if track_id is not None:
            assert isinstance(track_id, int) or isinstance(
                track_id, np.int64
            ), "Invalid track_id. It must be a integer number"
            self._track = np.int64(track_id)
        else:
            self._track = None

        if descr is not None:
            msg = "Invalid descriptor. It must be a numpy array with lenght of 128 or 256."
            assert isinstance(descr, np.ndarray), msg
            if len(descr.shape) == 1:
                if descr.shape[0] not in [128, 256]:
                    raise AssertionError(msg)
                else:
                    descr = descr.reshape(-1, 1)
            elif descr.shape[0] in [128, 256] and descr.shape[1] == 1:
                descr = descr.reshape(-1, 1)
            elif descr.shape[1] not in [128, 256]:
                raise AssertionError(msg)
            self._descr = np.float32(descr)
        else:
            self._descr = None

        if score is not None:
            assert isinstance(score, float) or isinstance(
                score, np.float32
            ), "Invalid score value. It must be a floating number of type np.float32"
            self._score = np.float32(score)
        else:
            self._score = None

        if epoch is not None:
            msg = "Invalid input argument epoch. It must be an integer number."
            try:
                epoch = np.int64(epoch)
            except:
                raise ValueError(msg)
            assert isinstance(epoch, np.int64), msg
            self.epoch = epoch
        else:
            self.epoch = None

    @property
    def x(self) -> np.float32:
        "Get x coordinate of the feature"
        return np.float32(self._x)

    @property
    def y(self) -> np.float32:
        "Get x coordinate of the feature"
        return np.float32(self._y)

    @property
    def xy(self) -> np.ndarray:
        "Get xy coordinates of the feature as 1x2 numpy array"
        return np.array([self._x, self._y], dtype=np.float32).reshape(1, 2)

    @property
    def track_id(self) -> np.int64:
        "Get x coordinate of the feature"
        if self._track is not None:
            return np.int64(self._track)
        else:
            logging.warning("Track id is not available")
            return None

    @property
    def descr(self) -> np.ndarray:
        """Get descriptor as mx1 numpy array (note that this is a column array)"""
        if self._descr is not None:
            return np.float32(self._descr)
        else:
            logging.warning("Descriptor is not available")
            return None

    @property
    def score(self) -> np.float32:
        """Get score"""
        if self._score is not None:
            return np.float32(self._score)
        else:
            logging.warning("Score is not available")
            return None


class Features:
    def __init__(self):
        self._values = {}
        self._last_id = -1
        self._iter = 0
        self._descriptor_size = 256

    def __len__(self) -> int:
        """
        __len__ Get number of features stored in Features object

        Returns:
            int: number of features
        """
        return self.num_features

    def __getitem__(self, track_id: np.int64) -> Feature:
        """
        __getitem__ Get Feature object by calling Features instance with [] based on track_id (e.g., features[track_id] to get first Feature object)

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
    def num_features(self):
        """
        num_features Number of features stored in Features object
        """
        return len(self._values)

    @property
    def last_track_id(self):
        """
        last_track_id Track_id of the last features stored in Features object
        """
        return self._last_id

    def append_feature(self, new_feature: Feature) -> None:
        """
        append_feature append a single Feature object to Features.

        Args:
            new_feature (Feature): Feature object to be appended. It must contain at least the x and y coordinates of the keypoint.
        """
        assert isinstance(
            new_feature, Feature
        ), "Invalid input feature. It must be Feature object"
        self._last_id += 1
        self._values[self._last_id] = new_feature
        if new_feature.descr is not None:
            if len(self) > 0:
                assert (
                    self._descriptor_size == new_feature.descr.shape[0]
                ), "Descriptor size of the new feature does not match with that of the existing feature"
            else:
                self._descriptor_size = new_feature.descr.shape[0]

    def set_last_track_id(self, last_track_id: np.int64) -> None:
        """
        set_last_track_id set track_id of last feature to a custom value

        Args:
            last_track_id (np.int64): track_id to set.
        """
        try:
            last_id = np.int64(last_track_id)
        except:
            raise ValueError(
                "Invalid input argument last_track_id. It must be an integer number."
            )
        self._last_id = last_id

    def append_features_from_numpy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        descr: np.ndarray = None,
        scores: np.ndarray = None,
        track_ids: List[np.int64] = None,
        epoch: np.int64 = None,
    ) -> None:
        """
        append_features_from_numpy append new features to Features object, starting from numpy arrays of x and y coordinates, descriptors and scores.

        Args:
            x (np.ndarray): nx1 numpy array containing x coordinates of all keypoints
            y (np.ndarray): nx1 numpy array containing y coordinates of all keypoints
            descr (np.ndarray, optional): mxn numpy array containing the descriptors of all the features (where m is the dimension of the descriptor that can be either 128 or 256). Defaults to None.
            scores (np.ndarray, optional):  nx1 numpy array containing scores of all keypoints. Defaults to None.
            track_ids (List[int]): Sorted list containing the track_id of each point to be added to Features object. Default to None.
            epoch (np.int64, optional): Epoch in which the incoming features are detected (or belongs to). Defaults to None.
        """
        assert isinstance(x, np.ndarray), "invalid type of x vector"
        assert isinstance(y, np.ndarray), "invalid type of y vector"
        assert descr.shape[0] in [
            128,
            256,
        ], "invalid shape of the descriptor array. It must be of size mxn (m: descriptor size [128, 256], n: number of features"

        if not np.any(x):
            logging.warning("Empty input feature arrays. Nothing done.")
            return None

        if descr is not None:
            if len(self) > 0:
                assert (
                    self._descriptor_size == descr.shape[0]
                ), "Descriptor size of the new feature does not match with that of the existing feature"
            else:
                self._descriptor_size = descr.shape[0]

        if epoch is not None:
            msg = "Invalid input argument epoch. It must be an integer number."
            try:
                epoch = np.int64(epoch)
            except:
                raise ValueError(msg)
            assert isinstance(epoch, np.int64), msg
            self.epoch = epoch

        xx = x.flatten()
        yy = y.flatten()

        if track_ids is None:
            ids = range(self._last_id + 1, self._last_id + len(xx) + 1)
        else:
            assert isinstance(
                track_ids, list
            ), "Invalid track_ids input. It must be a list of integers of the same size of the input arrays."
            assert len(track_ids) == len(
                xx
            ), "invalid size of track_id input. It must be a list of the same size of the input arrays."

            try:
                for id in track_ids:
                    if id in list(self._values.keys()):
                        msg = f"Feature with track_id {id} is already present in Features object. Ignoring input track_id and assigning progressive track_ids."
                        logging.error(msg)
                        raise ValueError(msg)
                ids = track_ids
            except ValueError:
                ids = range(self._last_id + 1, self._last_id + len(xx) + 1)

        if descr is not None:
            descr = np.float32(descr.T)
        else:
            descr = [None for _ in range(len(xx))]
        if scores is not None:
            scores = np.float32(scores.squeeze())
        else:
            scores = [None for _ in range(len(xx))]

        for id, x, y, d, s in zip(ids, xx, yy, descr, scores):
            self._values[id] = Feature(
                x,
                y,
                track_id=id,
                descr=d,
                score=s,
                epoch=epoch,
            )
            self._last_id = id

    def to_numpy(
        self,
        get_descr: bool = False,
        get_score: bool = False,
    ) -> dict:
        """
        to_numpy Get all keypoints (with, optionally, descriptors and scores) stacked as numpy arrays.

        Args:
            get_descr (bool, optional): get descriptors as mxn array. Defaults to False.
            get_score (bool, optional): get scores as nx1 array. Defaults to False.

        Returns:
            dict: dictionary containing the following keys (depending on the input arguments): ["kpts", "descr", "scores"]
        """
        kpts = np.empty((len(self), 2), dtype=np.float32)
        for i, v in enumerate(self._values.values()):
            kpts[i, :] = np.float32(v.xy)

        if get_descr and get_score:
            descr = self.descr_to_numpy()
            scores = self.scores_to_numpy()
            return {"kpts": kpts, "descr": descr, "scores": scores}
        elif get_descr:
            descr = self.descr_to_numpy()
            return {"kpts": kpts, "descr": descr}
        else:
            return {"kpts": kpts}

    def kpts_to_numpy(self) -> np.ndarray:
        """
        kpts_to_numpy Get all keypoints coordinates stacked as a nx2 numpy array.

        Returns:
            np.ndarray: nx2 numpy array containing xy coordinates of all keypoints
        """
        kpts = np.empty((len(self), 2))
        for i, v in enumerate(self._values.values()):
            kpts[i, :] = v.xy
        return np.float32(kpts)

    def descr_to_numpy(self) -> np.ndarray:
        """
        descr_to_numpy Get all descriptors stacked as a mxn (m is the descriptor size [128 or 256]) numpy array.

        Returns:
            np.ndarray: mxn numpy array containing the descriptors of all the features (where m is the dimension of the descriptor that can be either 128 or 256)
        """
        assert any(
            [self._values[i].descr is not None for i in list(self._values.keys())]
        ), "Descriptors non availble"
        descr = np.empty((self._descriptor_size, len(self)), dtype=np.float32)
        for i, v in enumerate(self._values.values()):
            descr[:, i : i + 1] = v.descr.reshape(-1, 1)
        return np.float32(descr)

    def scores_to_numpy(self) -> np.ndarray:
        """
        scores_to_numpy Get all scores stacked as a nx1 numpy array.

        Returns:
            np.ndarray: nx1 array with scores
        """
        assert any(
            [self._values[i].score is not None for i in list(self._values.keys())]
        ), "Scores non availble"
        score = np.empty(len(self), dtype=np.float32)
        for i, v in enumerate(self._values.values()):
            score[i] = v.score
        return np.float32(score)

    def get_track_ids(self) -> Tuple[np.int64]:
        """
        get_track_it Get a ordered tuple of track_id of all the features

        Returns:
            tuple: tuple of size (n,) with track_ids
        """
        return tuple([np.int64(x) for x in self._values.keys()])

    def get_features_as_dict(self, get_track_id: bool = False) -> dict:
        """
        get_features_as_dict Get a dictionary with keypoints, descriptors and scores, organized for SuperGlue

        Args:
            get_track_id (bool, optional): get a tuple with the track_id of all the features as an additionally dictionary key ["track_id"]. Defaults to False.

        Returns:
            dict: dictionary containing the following keys (depending on the input arguments): ["keypoints0", "descriptors0", "scores0"]
        """

        dict = {
            "keypoints0": self.kpts_to_numpy(),
            "descriptors0": self.descr_to_numpy(),
            "scores0": self.scores_to_numpy(),
        }
        if get_track_id:
            dict["track_id"] = self.get_track_ids()

        return dict

    def reset_fetures(self):
        """Reset Features instance"""
        self._values = {}
        self._last_id = -1
        self._iter = 0

    def filter_feature_by_mask(
        self, inlier_mask: List[bool], verbose: bool = False
    ) -> None:
        """
        delete_feature_by_mask Keep only inlier features, given a mask array as a list of boolean values. Note that this function does NOT take into account the track_id of the features! Inlier mask must have the same lenght as the number of features stored in the Features instance.

        Args:
            inlier_mask (List[bool]): boolean mask with True value in correspondance of the features to keep. inlier_mask must have the same length as the total number of features.
            verbose (bool): log number of filtered features. Defaults to False.
        """
        assert np.array_equal(
            inlier_mask, inlier_mask.astype(bool)
        ), "Invalid type of input argument for inlier_mask. It must be a boolean vector with the same lenght as the number of features stored in the Features object."
        assert len(inlier_mask) == len(
            self
        ), "Invalid shape of input argument for inlier_mask. It must be a boolean vector with the same lenght as the number of features stored in the Features object."

        feat_idx = list(self._values.keys())
        indexes = [feat_idx[i] for i, x in enumerate(inlier_mask) if x]
        self.filter_feature_by_index(indexes, verbose=verbose)

    def filter_feature_by_index(
        self, indexes: List[np.int64], verbose: bool = False
    ) -> None:
        """
        delete_feature_by_mask Keep only inlier features, given a list of index (int values) of the features to keep.

        Args:
            inlier_mask (List[int]): List with the index of the features to keep.
            verbose (bool): log number of filtered features. Defaults to False.

        """
        new_dict = {k: v for k, v in self._values.items() if v.track_id in indexes}
        if verbose:
            logging.info(
                f"Features filtered: {len(self)-len(new_dict)}/{len(self)} removed. New features size: {len(new_dict)}."
            )
        last_id = list(new_dict.keys())[-1]
        self._values = new_dict
        self._last_id = last_id

    def get_feature_by_index(self, indexes: List[np.int64]) -> dict:
        """
        get_feature_by_index Get inlier features, given a list of index (int values) of the features to keep.

        Args:
            indexes (List[int]): List with the index of the features to keep.
            verbose (bool, optional): log number of filtered features. Defaults to False.

        Returns:
            dict: dictionary containing the selected features with track_id as keys and Feature object as values {track_id: Feature}
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
        kpts = self.kpts_to_numpy()
        np.savetxt(
            path, kpts, fmt=fmt, delimiter=delimiter, newline="\n", header=header
        )

    def save_as_pickle(self, path: Union[str, Path]) -> True:
        """Save keypoints in as pickle file"""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    """Test classes"""

    from src.icepy.utils import setup_logger

    setup_logger()

    width, height = 6000, 4000
    n_feat = 10000
    x = np.random.randint(0, width, (n_feat, 1))
    y = np.random.randint(0, height, (n_feat, 1))
    kpts = np.concatenate((x, y), axis=1)
    descr = np.random.rand(256, n_feat)
    scores = np.random.rand(n_feat, 1)

    rep_times = 1

    features_new = Features()
    for _ in range(rep_times):
        t0 = time.time()
        features_new.append_features_from_numpy(x, y, descr)
        t1 = time.time()
        logging.info(
            f"Append features from numpy array to dict of Feature objects: Elapsed time {t1-t0:.4f} s"
        )

    for _ in range(rep_times):
        t0 = time.time()
        out = features_new.to_numpy(get_descr=True)
        t1 = time.time()
        logging.info(f"Get kpt+descr: Elapsed time {t1-t0:.4f} s")

    # Test iterable
    print(next(features_new).xy)
    print(next(features_new).xy)
    # for f in features_new:
    #     print(f.xy)

    print("Done")
