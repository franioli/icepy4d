import numpy as np
import logging
import importlib

from typing import List, Union, Tuple
from pathlib import Path

""" Calibration """


def read_opencv_calibration(
    path: Union[str, Path], verbose: bool = False
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Read camera internal orientation from file and return them.

    The file must contain the full K matrix and distortion vector,
    according to OpenCV standards, and organized in one line, as follow:
    width height fx 0. cx 0. fy cy 0. 0. 1. k1, k2, p1, p2, [k3, [k4, k5, k6
    Values must be float (include the . after integers) and divided by a
    white space.

    Parameters:
    - path: Union[str, Path]
        Path to the calibration file.
    - verbose: bool (optional)
        Verbosity flag for additional logging.

    Returns:
    Tuple[float, float, np.ndarray, np.ndarray]: (w, h, K, dist)
        w: float
            Image width.
        h: float
            Image height.
        K: np.ndarray
            Camera intrinsic matrix.
        dist: np.ndarray
            Distortion vector.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError("Calibration file does not exist.")

    with open(path, "r") as f:
        data = np.loadtxt(f)
        w = data[0]
        h = data[1]
        K = data[2:11].astype(float).reshape(3, 3, order="C")

        if len(data) == 15:
            if verbose:
                logging.info("Using OPENCV camera model.")
            dist = data[11:15].astype(float)
        elif len(data) == 16:
            if verbose:
                logging.info("Using OPENCV camera model + k3")
            dist = data[11:16].astype(float)
        elif len(data) == 19:
            if verbose:
                logging.info("Using FULL OPENCV camera model")
            dist = data[11:19].astype(float)
        else:
            raise ValueError(
                "Invalid intrinsics data. Calibration file must be formatted as follows:\nwidth height fx 0. cx 0. fy cy 0. 0. 1. k1, k2, p1, p2, [k3, [k4, k5, k6"
            )

    return w, h, K, dist


class Calibration:
    """"""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._K = None
        self._dist = None
        self._w = None
        self._h = None

        assert self.path.exists(), "Calibration file does not exist."

        match self.path.suffix:
            case ".txt":
                self._read_opencv()
            case ".json":
                self._read_json()
            case ".xml":
                self._read_xml()

    @property
    def K(self):
        return self._K

    @property
    def dist(self):
        return self._dist

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    def _read_opencv(self):
        self._w, self._h, self._K, self._dist = read_opencv_calibration(self.path)

    def to_camera(self):
        assert self._K is not None, "Calibration file not read."
        cam = importlib.import_module("icepy4d.classes.camera")
        return cam.Camera(self._K, self._dist, self._w, self._h)


if __name__ == "__main__":
    calib_reader = Calibration("data/calib/p1.txt")

    cam = calib_reader.to_camera()

    print("Done.")
