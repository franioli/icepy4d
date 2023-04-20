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


import numpy as np
import pandas as pd
import logging

from typing import List, Tuple, Union
from pathlib import Path


class Targets:
    """
    Class to store Target information, including image coordinates and object coordinates
    Targets are stored as numpy arrays

    Attributes:
        im_coor (List[np.ndarray]): A list of nx2 numpy arrays, where each numpy array contains the xy coordinates of the target projections on each image.
        obj_coor (np.ndarray): A nx3 numpy array containing XYZ object coordinates (can be empty).
    """

    def __init__(
        self,
        im_file_path: List[Union[str, Path]] = None,
        obj_file_path: Union[str, Path] = None,
    ) -> None:
        """
        Initialize a new instance of the Targets class.

        Args:
            im_file_path (List[Union[str, Path]], optional): A list of file paths to read image coordinates from. Defaults to None.
            obj_file_path (Union[str, Path], optional): A file path to read object coordinates from. Defaults to None.
        """
        self.reset_targets()

        # If im_coord_path is provided, read image coordinates from file
        if im_file_path is not None:
            for cam, path in enumerate(im_file_path):
                self.read_im_coord_from_txt(cam, path)
        if obj_file_path is not None:
            self.read_obj_coord_from_txt(obj_file_path)

    def __len__(self) -> int:
        """
        Return the total number of features stored.

        Returns:
            int: The total number of features stored.
        """
        return len(self.im_coor)

    def reset_targets(self) -> None:
        """Reset Target instance to empy list and None objects"""
        self.im_coor = []
        self.obj_coor = None

    def get_im_coord(self, cam_id=None) -> np.ndarray:
        """
        Return the image coordinates as a numpy array.

        Args:
            cam_id (int, optional): The camera ID to return the image coordinates for. If not provided, returns the list with the projections on all the cameras. Defaults to None.

        Returns:
            np.ndarray: A numpy array containing the image coordinates.
        """
        if cam_id is None:
            return np.float32(self.im_coor)
        else:
            # Return object coordinates as numpy array
            return np.float32(self.im_coor[cam_id].iloc[:, 1:])

    def get_obj_coord(self) -> np.ndarray:
        """
        Return the object coordinates as a numpy array.

        Returns:
            np.ndarray: A numpy array containing the object coordinates.
        """
        # Return object coordinates as numpy array
        return np.float32(self.obj_coor.iloc[:, 1:])

    def append_obj_cord(self, new_obj_coor) -> None:
        """
        Append new object coordinates to the current object coordinates.

        Args:
            new_obj_coor (np.ndarray): The new object coordinates to append.

        TODO: add checks on input dimensions
        """
        if self.obj_coor is None:
            self.obj_coor = new_obj_coor
        else:
            self.obj_coor = np.append(self.obj_coor, new_obj_coor, axis=0)

    def get_target_labels(self, cam_id=None) -> List[str]:
        """
        Returns target labels as a list.

        Args:
            cam_id (int, optional): Numeric id of the camera for which the target labels are requested.

        Returns:
            List: Target labels as a list.
        """
        if cam_id is not None:
            return list(self.im_coor[cam_id].label)
        else:
            return [list(self.im_coor[x].label) for x, _ in enumerate(self.im_coor)]

    def get_image_coor_by_label(
        self,
        labels: List[str],
        cam_id: int,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        get_image_coor_by_label Get image coordinates of the targets on the images given a list of target labels

        Args:
            labels (List[str]): List containing the targets' label to extract
            cam_id (int): numeric (integer) id of the camera for which the image coordinates are asked.

        Raises:
            ValueError: No targets are found.

        Returns:
            Tuple[np.ndarray, List[str]]: Tuple cointaining a numpy nx2 array with the coordinates of the selected (and valid) targets and a list with the labels of the valid targets
        """

        coor = []
        labels_valid = []
        for lab in labels:
            selected = self.im_coor[cam_id][self.im_coor[cam_id]["label"] == lab]
            if not selected.empty:
                coor.append(selected.iloc[:, 1:].to_numpy())
                labels_valid.append(lab)
            else:
                logging.warning(
                    f"Warning: target {lab} is not present on camera {cam_id}."
                )

        # If at least one target was found, concatenate arrays to return nx3 array containing world coordinates
        if coor:
            return np.concatenate(coor, axis=0), labels_valid
        else:
            msg = "No targets with the provided labels found."
            logging.error(msg)
            raise ValueError(msg)

    def get_object_coor_by_label(
        self,
        labels: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        get_object_coor_by_label Get object coordinates of the targets on the images given a list of target labels

        Args:
            labels (List[str]): List containing the targets' label to extract

        Raises:
            ValueError: No targets are found.

        Returns:
            Tuple[np.ndarray, List[str]]: Tuple cointaining a nx3 array containing object coordinates of the selected (and valid) targets and a list with the labels of the valid targets
        """
        coor = []
        labels_valid = []
        for lab in labels:
            selected = self.obj_coor[self.obj_coor["label"] == lab]
            if not selected.empty:
                coor.append(selected.iloc[:, 1:].to_numpy())
                labels_valid.append(lab)
            else:
                logging.warning(f"Warning: target {lab} is not present.")

        # If at least one target was found, concatenate arrays to return nx3 array containing world coordinates
        if coor:
            return np.concatenate(coor, axis=0), labels_valid
        else:
            msg = "No targets with the provided labels found."
            logging.error(msg)
            raise ValueError(msg)

    def read_im_coord_from_txt(
        self,
        camera_id,
        path,
        delimiter: str = ",",
        header: int = 0,
    ) -> None:
        """
        Read target image coordinates from a .txt file and store them in a pandas DataFrame organized as follows:

            - One line per target
            - label, then first x coordinate, then y coordinate
            - Coordinates separated by a delimiter (default: ',')
            e.g.
            # label,x,y
            target_1,1000,2000
            target_2,2000,3000

        Args:
            camera_id (int): The numeric (integer) ID of the camera to which the target coordinates belong.
            path (Union[str, Path]): The path to the input file.
            delimiter (str, optional): The delimiter used in the input file (default: ',').
            header (int, optional): The row number to use as the column names. Default is 0 (the first row).

        Raises:
            FileNotFoundError: If the input path does not exist.

        Returns:
            None
        """
        assert isinstance(
            camera_id, int
        ), "Missing or invalid camera id. Impossible to assign the target coordinates to the correct camera"

        path = Path(path)
        if not path.exists():
            msg = f"Error: Input path {path} does not exist."
            logging.error(msg)
            raise FileNotFoundError(msg)

        data = pd.read_csv(path, sep=delimiter, header=header)

        self.im_coor.insert(camera_id, data)

    def read_obj_coord_from_txt(
        self,
        path=None,
        delimiter: str = ",",
        header: int = 0,
    ) -> None:
        """
        Read target object coordinates from a .txt file and store them in a pandas DataFrame organized as follows:

            - One line per target
            - label, then first x coordinate, then y coordinate
            - Coordinates separated by a delimiter (default: ',')
            e.g.
            # label,X,Y,Z
            target_1,1000,2000,3000
            target_1,2000,3000,3000

        Args:
            path (Union[str, Path]): The path to the input file.
            delimiter (str, optional): The delimiter used in the input file (default: ',').
            header (int, optional): The row number to use as the column names. Default is 0 (the first row).

        Raises:
            FileNotFoundError: If the input path does not exist.

        Returns:
            None.
        """

        path = Path(path)
        if not path.exists():
            msg = f"Error: Input path {path} does not exist."
            logging.error(msg)
            raise FileNotFoundError(msg)

        data = pd.read_csv(path, sep=delimiter, header=header)

        self.append_obj_cord(data)

    # def save_as_txt(self, path=None, fmt='%i', delimiter=',', header='x,y'):
    #     ''' Save keypoints in a .txt file '''
    #     if path is None:
    #         print("Error: missing path argument.")
    #         return
    #     # if not Path(path).:
    #     #     print('Error: invalid input path.')
    #     #     return
    #     np.savetxt(path, self.kpts, fmt=fmt, delimiter=delimiter,
    #                newline='\n', header=header)


if __name__ == "__main__":
    """Test classes"""

    from icepy4d.utils.initialization import parse_yaml_cfg, Inizialization

    CFG_FILE = "config/config_2021_1.yaml"
    cfg = parse_yaml_cfg(CFG_FILE)

    init = Inizialization(cfg)
    init.inizialize_icepy()
    cams = init.cams
    images = init.images
    targets = init.targets

    epoch = 0
    tt = cfg.georef.targets_to_use
    obj, labels = targets[epoch].get_object_coor_by_label(["F1", "F2"])
    image_points, labels = targets[epoch].get_image_coor_by_label(tt, cam_id=0)
