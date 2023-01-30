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

from .camera import Camera


class Features:
    """
    Class to store matched features, descriptors and scores
    Features are stored as numpy arrays:
        Features.kpts: nx2 array of features location
        Features.descr: mxn array of descriptors(note that descriptors are stored columnwise)
        Features.score: nx1 array with feature score"""

    def __init__(
        self,
        logging: logging = None,
    ):
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
    pass
