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

if __name__ == "__main__":
    from src.icepy.base_classes.camera import Camera
else:
    from .camera import Camera


class Feature:
    def __init__(
        self,
        x: Union[float, int],
        y: Union[float, int],
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
        self._x = x
        self._y = y

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
            elif descr.shape[1] not in [128, 256]:
                raise AssertionError(msg)
            self._descr = descr
        else:
            self._descr = None

        if score is not None:
            self._score = score
        else:
            self._score = None

    @property
    def x(self) -> float:
        "Get x coordinate of the feature"
        return float(self._x)

    @property
    def y(self) -> float:
        "Get x coordinate of the feature"
        return float(self._y)

    @property
    def xy(self) -> np.ndarray:
        "Get xy coordinates of the feature as 1x2 numpy array"
        return np.array([self._x, self._y], dtype=float).reshape(1, 2)

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

            return self._descr
        else:
            logging.warning("Descriptor is not available")
            return None

    @property
    def score(self) -> float:
        """Get score"""
        if self._score is not None:
            return float(self._score)
        else:
            logging.warning("Score is not available")
            return None


class Features_new:
    def __init__(self):
        self._values = {}
        self._increm_id = 0

    def __len__(self) -> int:
        """
        __len__ _summary_

        Returns:
            int: _description_
        """
        return len(self._values)

    def __getitem__(self, track_id: int) -> str:
        """
        __getitem__ _summary_

        Args:
            track_id (int): _description_

        Returns:
            str: _description_
        """
        if track_id in list(self._values.keys()):
            return self._values[track_id]
        else:
            logging.warning(f"Feature with track id {track_id} not available.")
            return None

    # def __iter__(self):
    #     self._elem = 0
    #     return self

    # def __next__(self):
    #     while self._elem < len(self):
    #         file = self.files[self._elem]
    #         self._elem += 1
    #         return file
    #     else:
    #         self._elem
    #         raise StopIteration

    def append_feature(self, new_feature: Feature):
        assert isinstance(
            new_feature, Feature
        ), "Invalid input feature. It must be Feature object"
        self._values[self._increm_id] = new_feature
        self._increm_id += 1

    def append_features_from_numpy_array(self, x, y):
        xx = x.flatten()
        yy = y.flatten()
        ids = range(self._increm_id, self._increm_id + len(xx))
        for x, y, id in zip(xx, yy, ids):
            self._values[id] = Feature(x, y)

        self._increm_id = self._increm_id + len(xx)

    def reset_fetures(self):
        """
        Reset Feature instance to None Objects
        """
        self._values = {}
        self._increm_id = 0


class Features:
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
        # TODO: write description
        """Remove outliers features
        Parameters
        - ---------
        new_features: TYPE
            DESCRIPTION.

        Returns
        - ------
        None.
        """
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
        # TODO: check correct shape of inputs.

        if self.kpts is None:
            self.kpts = new_features["kpts"]
            self.descr = new_features["descr"]
            self.score = new_features["score"]
        else:
            self.kpts = np.append(self.kpts, new_features["kpts"], axis=0)
            self.descr = np.append(self.descr, new_features["descr"], axis=1)
            self.score = np.append(self.score, new_features["score"], axis=0)

    def save_as_pickle(self, path=None):
        """Save keypoints in a .txt file"""
        if path is None:
            print("Error: missing path argument.")
            return
        # if not Path(path).:
        #     print('Error: invalid input path.')
        #     return
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_as_txt(self, path=None, fmt="%i", delimiter=",", header="x,y"):
        """Save keypoints in a .txt file"""
        if path is None:
            print("Error: missing path argument.")
            return
        # if not Path(path).:
        #     print('Error: invalid input path.')
        #     return
        np.savetxt(
            path, self.kpts, fmt=fmt, delimiter=delimiter, newline="\n", header=header
        )


if __name__ == "__main__":
    """Test classes"""

    from src.icepy.utils import setup_logger

    setup_logger()

    width, height = 6000, 4000
    n_feat = 100000
    x = np.random.randint(0, width, (n_feat, 1))
    y = np.random.randint(0, height, (n_feat, 1))
    kpts = np.concatenate((x, y), axis=1)
    descr = np.random.rand(256, n_feat)
    scores = np.random.rand(n_feat, 1)

    t0 = time.time()
    features = Features()
    features.append_features(
        {
            "kpts": kpts,
            "descr": descr,
            "score": scores,
        }
    )
    t1 = time.time()
    logging.info(f"Append features as numpy array: elapsed time {t1-t0:.4f} s")

    t0 = time.time()
    features = Features_new()
    for i in range(len(x)):
        features.append_feature(Feature(x[i, 0], y[i, 0]))
    t1 = time.time()
    logging.info(
        f"Append features as single Feature objects: Elapsed time {t1-t0:.4f} s"
    )

    t0 = time.time()
    features = Features_new()
    features.append_features_from_numpy_array(x, y)
    t1 = time.time()
    logging.info(
        f"Append features from numpy array to dict of Feature objects: Elapsed time {t1-t0:.4f} s"
    )

    features[0].descr

    print("Done")
