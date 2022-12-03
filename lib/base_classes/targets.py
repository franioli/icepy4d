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

import os
import cv2
import pickle
import numpy as np
import pandas as pd
import logging

# import exifread
from typing import List, Union, Tuple
from scipy import linalg
from pathlib import Path

from lib.import_export.importing import read_opencv_calibration


class Targets:
    """
    Class to store Target information, including image coordinates and object coordinates
    Targets are stored as numpy arrays:
        Targets.im_coor: [nx2] List of array of containing xy coordinates of the
                        target projections on each image
        Targets.obj_coor: nx3 array of XYZ object coordinates(it can be empty)
    """

    def __init__(
        self,
        cam_id=None,
        im_file_path=None,
        obj_file_path=None,
        logger: logging = None,
    ):
        self.reset_targets()

        # If im_coord_path is provided, read image coordinates from file
        if im_file_path is not None:
            for cam, path in enumerate(im_file_path):
                self.read_im_coord_from_txt(cam, path)
        if obj_file_path is not None:
            self.read_obj_coord_from_txt(obj_file_path)

    def __len__(self):
        """Get total number of featues stored"""
        return len(self.im_coor)

    def reset_targets(self):
        """
        Reset Target instance to empy list and None objects
        """
        self.im_coor = []
        self.obj_coor = None

    def get_im_coord(self, cam_id=None):
        """
        Return image coordinates as numpy array
        If numeric camera id(integer) is provided, the function returns the
        image coordinates in that camera, otherwise the list with the projections on all the cameras is returned.
        """
        if cam_id is None:
            return np.float32(self.im_coor)
        else:
            # Return object coordinates as numpy array
            return np.float32(self.im_coor[cam_id].iloc[:, 1:])

    def get_obj_coord(self):
        # Return object coordinates as numpy array
        return np.float32(self.obj_coor.iloc[:, 1:])

    def append_obj_cord(self, new_obj_coor):
        # TODO: add check on dimension and add description
        if self.obj_coor is None:
            self.obj_coor = new_obj_coor
        else:
            self.obj_coor = np.append(self.obj_coor, new_obj_coor, axis=0)

    def get_target_labels(self, labels: List[str], cam_id=None):
        """ """
        pass

    def extract_image_coor_by_label(
        self,
        labels: List[str],
        cam_id: int = None,
    ) -> np.ndarray:
        """
        Return image coordinates of the targets on the images given a list of target labels
        """
        # try:
        coor = []
        for lab in labels:
            if cam_id is not None:
                selected = self.im_coor[cam_id][self.im_coor[cam_id]["label"] == lab]
                if not selected.empty:
                    coor.append(np.float32(selected.iloc[:, 1:]))
                else:
                    print(f"Warning: target {lab} is not present.")
            else:
                print("provide cam id")
                return None

        return np.concatenate(coor, axis=0)
        # except:
        #     pass

    def extract_object_coor_by_label(
        self,
        labels: List[str],
    ) -> np.ndarray:
        """
        Return object coordinates of the targets on the images given a list of target labels
        """
        coor = []
        for lab in labels:
            selected = self.obj_coor[self.obj_coor["label"] == lab]
            if not selected.empty:
                coor.append(np.float32(selected.iloc[:, 1:]))
            else:
                print(f"Warning: target {lab} is not present.")

        return np.concatenate(coor, axis=0)

    def read_im_coord_from_txt(
        self,
        camera_id=None,
        path=None,
        delimiter: str = ",",
        header: int = 0,
        column_names: List[str] = None,
        from_metashape: bool = True,
    ):
        """
        Read image target image coordinates from .txt file in a pandas dataframe
        organized as follows:
            - One line per target
            - label, then first x coordinate, then y coordinate
            - Coordinates separated by a delimiter(default ',')
            e.g.
            # label,x,y
            target_1,1000,2000
            target_2,2000,3000
        """
        if camera_id is None:
            print(
                "Error: missing camera id. Impossible to assign the target\
                  coordinates to the correct camera"
            )
            return
        if path is None:
            print("Error: missing path argument.")
            return
        path = Path(path)
        if not path.exists():
            print("Error: Input path does not exist.")
            return
        data = pd.read_csv(path, sep=delimiter, header=header)

        # subtract 0.5 px to image coordinates (metashape image RS)
        if from_metashape:
            data.x = data.x - 0.5
            data.y = data.y - 0.5

        self.im_coor.insert(camera_id, data)

    def read_obj_coord_from_txt(
        self,
        path=None,
        delimiter: str = ",",
        header: int = 0,
        column_names: List[str] = None,
    ):
        """
        Read image target image coordinates from .txt file in a pandas dataframe
        organized as follows:
            - One line per target
            - label, then first x coordinate, then y coordinate
            - Coordinates separated by a delimiter(default ',')
            e.g.
            # label,X,Y,Z
            target_1,1000,2000,3000
            target_1,2000,3000,3000
        """
        if path is None:
            print("Error: missing path argument.")
            return
        path = Path(path)
        if not path.exists():
            print("Error: Input path does not exist.")
            return
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
    pass
