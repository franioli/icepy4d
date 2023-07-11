import importlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union
import platform

import numpy as np

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
    assert path.exists(), "Calibration file does not exist."
    assert path.suffix == ".txt", "Calibration file must be a .txt file."

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


def read_xml_calibration(
    path: Union[str, Path],
    format: str = "metashape",
) -> Tuple[float, float, np.ndarray, np.ndarray, datetime]:
    """
    NOT COMPLETED YET
    """

    path = Path(path)
    assert path.exists(), "Calibration file does not exist."
    assert path.suffix == ".xml", "Calibration file must be a .xml file."

    tree = ET.parse(path)
    root = tree.getroot()

    if format == "metashape":
        try:
            w = float(root.find("width").text)
            h = float(root.find("height").text)
            f = float(root.find("f").text)
            cx = float(root.find("cx").text)
            cy = float(root.find("cy").text)
            k1 = float(root.find("k1").text)
            k2 = float(root.find("k2").text)
            p1 = float(root.find("p1").text)
            p2 = float(root.find("p2").text)
            date_str = root.find("date").text
            date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            dist = np.array([k1, k2, p1, p2])
        except Exception as e:
            raise ValueError(
                "Unable to read xml calibration file as a Agisoft Metashape format. Check the file format."
            )

    elif format == "opencv":
        try:
            w = int(root.find("image_Width").text)
            h = int(root.find("image_Height").text)

            camera_matrix_data = root.find("Camera_Matrix").find("data").text
            camera_matrix_values = camera_matrix_data.split()
            K = np.array(camera_matrix_values, dtype=float).reshape(3, 3)

            distortion_data = root.find("Distortion_Coefficients").find("data").text
            distortion_values = distortion_data.split()
            dist = np.array(distortion_values, dtype=float).reshape(-1, 1)

            calibration_time_str = root.find("calibration_Time").text
            date = datetime.strptime(
                calibration_time_str.strip('"'), "%a %b %d %H:%M:%S %Y"
            )

        except Exception as e:
            raise ValueError(
                "Unable to read xml calibration file as a OpenCV format. Check the file format."
            )

    return w, h, K, dist, date


class Calibration:
    """"""

    def __init__(self, path: Union[str, Path], **kwargs) -> None:
        self.path = Path(path)
        self._K = None
        self._dist = None
        self._w = None
        self._h = None
        self._date = None

        assert self.path.exists(), "Calibration file does not exist."

        try:
            match self.path.suffix:
                case ".txt":
                    self._read_opencv()
                case ".xml":
                    fmt = kwargs.get("format", "metashape")
                    self._read_xml(format=fmt)
                # case ".json":
                #     self._read_json()
        except:
            # bakcwards compatibility with Python < 3.10
            if self.path.suffix == ".txt":
                self._read_opencv()
            elif self.path.suffix == ".xml":
                fmt = kwargs.get("format", "metashape")
                self._read_xml(format=fmt)
            # elif self.path.suffix == ".json":
            #     self._read_json()
            else:
                raise ValueError("Invalid calibration file format.")

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

    def _read_xml(self, format: str):
        self._w, self._h, self._K, self._dist, self._date = read_xml_calibration(
            self.path, format=format
        )

    def to_camera(self):
        assert self._K is not None, "Calibration file not read."
        cam = importlib.import_module("icepy4d.classes.camera")
        return cam.Camera(
            width=self._w,
            height=self._h,
            K=self._K,
            dist=self._dist,
        )


if __name__ == "__main__":
    calib_from_txt = Calibration("data/calib/p2.txt")
    cam0 = calib_from_txt.to_camera()

    calib_from_xml = Calibration("data/calib/35mm_280722_selfcal_all_metashape.xml")
    cam1 = calib_from_xml.to_camera()

    calib_from_xml = Calibration(
        "data/calib/35mm_280722_selfcal_all_opencv.xml", format="opencv"
    )
    cam2 = calib_from_xml.to_camera()

    print("Done.")
