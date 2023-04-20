import pickle

from typing import Union
from pathlib import Path
import logging

import icepy4d.classes as classes


class Solution:
    """
    Class for storing, saving, and reading ICEpy4D solution at one epoch

    Attributes:
        cameras (classes.Camera): The dictionary of camera parameters
        images (classes.ImagesDict): The dictionary of images and their metadata
        features (classes.Features): The dictionary of feature points
        points (classes.Points): The dictionary of 3D points
    """

    def __init__(
        self,
        cameras: classes.Camera,
        images: classes.ImagesDict,
        features: classes.Features,
        points: classes.Points,
    ) -> None:
        """
        Initializes a Solution object with the provided data

        Args:
            cameras (classes.Camera): The dictionary of camera parameters
            images (classes.ImagesDict): The dictionary of images and their metadata
            features (classes.Features): The dictionary of feature points
            points (classes.Points): The dictionary of 3D points
        """

        self.cameras = cameras
        self.images = images
        self.features = features
        self.points = points

    def __iter__(self):
        """
        Returns an iterator over the four dictionaries of Solution object

        Yields:
            dict: The dictionary of camera parameters
            dict: The dictionary of images and their metadata
            dict: The dictionary of feature points
            dict: The dictionary of 3D points
        """
        yield self.cameras
        yield self.images
        yield self.features
        yield self.points

    def __hash__(self):
        """
        Computes the hash value of the Solution object

        Returns:
            int: The hash value of the Solution object
        """
        return hash((self.cameras, self.images, self.features, self.points))

    def save_solutions(self, path: Union[str, Path]) -> bool:
        """
        Saves the Solution object to a binary file

        Args:
            path (Union[str, Path]): The path to the binary file

        Returns:
            bool: True if the object was successfully saved to file, False otherwise
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            return True
        except:
            logging.error("Unable to save the Solution as Pickle object")
            return False

    @staticmethod
    def read_solution(path: Union[str, Path], ignore_errors: bool = False):
        """
        Loads a Solution object from a binary file

        Args:
            path (Union[str, Path]): The path to the binary file

        Returns:
            Solution: A Solution object
        """
        path = Path(path)
        if not ignore_errors:
            assert path.exists(), "Input path does not exists"

        try:
            with open(path, "rb") as inp:
                solution = pickle.load(inp)
            return solution
        except:
            logging.error("Unable to read Solution from pickle file")
            return None
