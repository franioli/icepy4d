import cv2
import numpy as np
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
                print("Using OPENCV camera model.")
            dist = data[11:15].astype(float)
        elif len(data) == 16:
            if verbose:
                print("Using OPENCV camera model + k3")
            dist = data[11:16].astype(float)
        elif len(data) == 19:
            if verbose:
                print("Using FULL OPENCV camera model")
            dist = data[11:19].astype(float)
        else:
            raise ValueError(
                "Invalid intrinsics data. Calibration file must be formatted as follows:\nwidth height fx 0. cx 0. fy cy 0. 0. 1. k1, k2, p1, p2, [k3, [k4, k5, k6"
            )

    return w, h, K, dist


""" Images """


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


def read_img(path, color=True, resize=[-1], crop=None):
    if color:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)

    if image is None:
        return None, None

    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    image = cv2.resize(image, (w_new, h_new))

    if crop:
        image = image[crop[1] : crop[3], crop[0] : crop[2]]

    return image, scales
