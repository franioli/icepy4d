import numpy as np
import logging

from typing import List, Union
from pathlib import Path

""" Calibration """


def read_opencv_calibration(path: Union[str, Path], verbose: bool = False):
    """
    Read camera internal orientation from file and return them.
    The file must contain the full K matrix and distortion vector,
    according to OpenCV standards, and organized in one line, as follow:
    width height fx 0. cx 0. fy cy 0. 0. 1. k1, k2, p1, p2, [k3, [k4, k5, k6
    Values must be float(include the . after integers) and divided by a
    white space.
    -------
    Returns: (w, h, K, dist)
    """
    path = Path(path)
    if not path.exists():
        raise ValueError("Calibration filed does not exist.")
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
