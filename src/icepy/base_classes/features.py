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

from typing import Union, List
from pathlib import Path


if __name__ == "__main__":
    from src.icepy.base_classes.camera import Camera
else:
    from .camera import Camera


class Feature:
    __slots__ = ("_x", "_y", "_track", "_descr", "_score")

    def __init__(
        self,
        x: float,
        y: float,
        track_id: int = None,
        descr: np.ndarray = None,
        score: float = None,
    ) -> None:
        """
        __init__ Create Feature object

        Args:
            x (Union[float, int]): x coordinate (as for OpenCV image coordinate system)
            y (Union[float, int]): y coordinate (as for OpenCV image coordinate system)
            track_id (int, optional): track_id. Defaults to None.
            descr (np.ndarray, optional): descriptor as a numpy array with 128 or 256 elements. Defaults to None.
            score (float, optional): score. Defaults to None.

        Raises:
            AssertionError: Invalid shape of the descriptor. It must be a numpy array with 128 or 256 elements.
        """
        self._x = np.float32(x)
        self._y = np.float32(y)

        if track_id is not None:
            assert isinstance(
                track_id, int
            ), "Invalid track_id. It must be a integer number"
            self._track = track_id
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
            self._score = score
        else:
            self._score = None

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
    def track_id(self) -> int:
        "Get x coordinate of the feature"
        if self._track is not None:
            return int(self._track)
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
        self._increm_id = 0
        self._iter = 0
        self._descriptor_size = 256

    def __len__(self) -> int:
        """
        __len__ _summary_

        Returns:
            int: _description_
        """
        return len(self._values)

    def __getitem__(self, track_id: int) -> Feature:
        """
        __getitem__ _summary_

        Args:
            track_id (int): _description_

        Returns:
            Feature: _description_
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

    def append_feature(self, new_feature: Feature) -> None:
        """
        append_feature append a single Feature object to Features.

        Args:
            new_feature (Feature): Feature object to be appended. It must contain at least the x and y coordinates of the keypoint.
        """
        assert isinstance(
            new_feature, Feature
        ), "Invalid input feature. It must be Feature object"
        self._values[self._increm_id] = new_feature
        if new_feature.descr is not None:
            if len(self) > 0:
                assert (
                    self._descriptor_size == new_feature.descr.shape[0]
                ), "Descriptor size of the new feature does not match with that of the existing feature"
            else:
                self._descriptor_size = new_feature.descr.shape[0]

        self._increm_id += 1

    def append_features_from_numpy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        descr: np.ndarray = None,
        scores: np.ndarray = None,
    ) -> None:
        """
        append_features_from_numpy append new features to Features object, starting from numpy arrays of x and y coordinates, descriptors and scores.

        Args:
            x (np.ndarray): nx1 numpy array containing x coordinates of all keypoints
            y (np.ndarray): nx1 numpy array containing y coordinates of all keypoints
            descr (np.ndarray, optional): mxn numpy array containing the descriptors of all the features (where m is the dimension of the descriptor that can be either 128 or 256). Defaults to None.
            scores (np.ndarray, optional):  nx1 numpy array containing scores of all keypoints. Defaults to None.
        """
        assert isinstance(x, np.ndarray), "invalid type of x vector"
        assert isinstance(y, np.ndarray), "invalid type of y vector"
        assert descr.shape[0] in [
            128,
            256,
        ], "invalid shape of the descriptor array. It must be of size mxn (m: descriptor size [128, 256], n: number of features"

        if descr is not None:
            if len(self) > 0:
                assert (
                    self._descriptor_size == descr.shape[0]
                ), "Descriptor size of the new feature does not match with that of the existing feature"
            else:
                self._descriptor_size = descr.shape[0]

        xx = x.flatten()
        yy = y.flatten()
        ids = range(self._increm_id, self._increm_id + len(xx))
        if descr is not None:
            descr = np.float32(descr.T)
        else:
            descr = [None for _ in range(len(xx))]
        if scores is not None:
            scores = np.float32(scores.squeeze())
        else:
            scores = [None for _ in range(len(xx))]
        for x, y, id, d, s in zip(xx, yy, ids, descr, scores):
            self._values[id] = Feature(x, y, track_id=id, descr=d, score=s)
        self._increm_id = self._increm_id + len(xx)

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

    def get_features_as_dict(self) -> dict:
        """
        get_features_as_dict Return a dictionary with keypoints, descriptors and scores, organized for SuperGlue

        Returns:
            dict: dictionary containing the following keys (depending on the input arguments): ["keypoints0", "descriptors0", "scores0"]
        """

        dict = {
            "keypoints0": self.kpts_to_numpy(),
            "descriptors0": self.descr_to_numpy(),
            "scores0": self.scores_to_numpy(),
        }
        return dict

    def reset_fetures(self):
        """Reset Features instance"""
        self._values = {}
        self._increm_id = 0
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
        indexes = [i for i, x in enumerate(inlier_mask) if x]
        self.filter_feature_by_index(indexes, verbose=verbose)

    def filter_feature_by_index(self, indexes: List[int], verbose: bool) -> None:
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
        self._values = new_dict

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


class Features_old:
    """
    Class to store matched features, descriptors and scores
    Features are stored as numpy arrays:
        Features.kpts: nx2 array of features location
        Features.descr: mxn array of descriptors(note that descriptors are stored columnwise)
        Features.score: nx1 array with feature score"""

    def __init__(self):
        self.reset_fetures()

    def __len__(self):
        """Get total number of featues stored"""
        return len(self.kpts)

    def reset_fetures(self):
        """
        Reset Feature instance to None Objects
        """
        self.kpts = None
        self.descr = None
        self.score = None

    def initialize_fetures(self, nfeatures=1, descr_size=256):
        """
        Inizialize Feature instance to numpy arrays,
        optionally for a given number of features and descriptor size(default is 256).
        """
        self.kpts = np.empty((nfeatures, 2), dtype=float)
        self.descr = np.empty((descr_size, nfeatures), dtype=float)
        self.score = np.empty(nfeatures, dtype=float)

    def get_keypoints(self):
        """Return keypoints as numpy array"""
        return np.float32(self.kpts)

    def get_descriptors(self):
        """Return descriptors as numpy array"""
        return np.float32(self.descr)

    def get_scores(self):
        """Return scores as numpy array"""
        return np.float32(self.score)

    def get_features_as_dict(self):
        """Return a dictionary with keypoints, descriptors and scores, organized for SuperGlue"""
        out = {
            "keypoints0": self.get_keypoints(),
            "descriptors0": self.get_descriptors(),
            "scores0": self.get_scores(),
        }
        return out

    def remove_outliers_features(self, inlier_mask):
        self.kpts = self.kpts[inlier_mask, :]
        self.descr = self.descr[:, inlier_mask]
        self.score = self.score[inlier_mask]

    def append_features(self, new_features):
        """
        Append new features to Features Class.
        Input new_features is a Dict with keys as follows:
            new_features['kpts']: nx2 array of features location
            new_features['descr']: mxn array of descriptors(note that descriptors are stored columnwise)
            new_features['score']: nx1 array with feature score
        """
        # Check dictionary keys:
        keys = ["kpts", "descr", "score"]
        if any(key not in new_features.keys() for key in keys):
            logging.error(
                'Invalid input dictionary. Check all keys ["kpts", "descr", "scores"] are present'
            )
            return self

        if self.kpts is None:
            self.kpts = new_features["kpts"]
            self.descr = new_features["descr"]
            self.score = new_features["score"]
        else:
            self.kpts = np.append(self.kpts, new_features["kpts"], axis=0)
            self.descr = np.append(self.descr, new_features["descr"], axis=1)
            self.score = np.append(self.score, new_features["score"], axis=0)

    def save_as_pickle(self, path):
        """Save keypoints in a .txt file"""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_as_txt(self, path, fmt="%i", delimiter=",", header="x,y"):
        """Save keypoints in a .txt file"""
        np.savetxt(
            path, self.kpts, fmt=fmt, delimiter=delimiter, newline="\n", header=header
        )


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

    # rep_times = 1

    # features = Features_old()
    # for _ in range(rep_times):
    #     t0 = time.time()
    #     features.append_features(
    #         {
    #             "kpts": kpts,
    #             "descr": descr,
    #             "score": scores,
    #         }
    #     )
    #     t1 = time.time()
    #     logging.info(f"Append features as numpy array: elapsed time {t1-t0:.4f} s")

    # features_new = Features()
    # for _ in range(rep_times):
    #     t0 = time.time()
    #     features_new.append_features_from_numpy(x, y, descr)
    #     t1 = time.time()
    #     logging.info(
    #         f"Append features from numpy array to dict of Feature objects: Elapsed time {t1-t0:.4f} s"
    #     )

    # for _ in range(rep_times):
    #     t0 = time.time()
    #     out = features_new.kpts_to_numpy()
    #     t1 = time.time()
    #     logging.info(f"Get xy coordinates: Elapsed time {t1-t0:.4f} s")

    # for _ in range(rep_times):
    #     t0 = time.time()
    #     out = features_new.descr_to_numpy()
    #     t1 = time.time()
    #     logging.info(f"Get descr: Elapsed time {t1-t0:.4f} s")

    # for _ in range(rep_times):
    #     t0 = time.time()
    #     out = features_new.to_numpy(get_descr=True)
    #     t1 = time.time()
    #     logging.info(f"Get kpt+descr: Elapsed time {t1-t0:.4f} s")

    # # Test iterable
    # print(next(features_new).xy)
    # print(next(features_new).xy)
    # # for f in features_new:
    # #     print(f.xy)

    print("Done")
