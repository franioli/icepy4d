import numpy as np
import cv2
import logging

from typing import List

from lib.base_classes.features import Features
from lib.base_classes.camera import Camera
from lib.base_classes.images import Image, ImageDS

from lib.matching.matching_base import MatchingAndTracking
from lib.sfm.triangulation import Triangulate
from lib.geometry import (
    estimate_pose,
    undistort_image,
    undistort_points,
    project_points,
)

logger = logging.getLogger(__name__)

class IncrementalReconstruction:
    """
     _summary_
    """
    
    def __init__(self, cameras: List[Camera], feautres: List[np.ndarray], preselection: str = "exaustive") -> None:
        """
        __init__ _summary_

        Args:
            cameras (List[Camera]): _description_
            feautres (List[np.ndarray]): _description_
            preselection (str, optional): _description_. Defaults to "exaustive".
        """
        self.cameras = cameras
        self.features = feautres
        self.preselection = preselection
        
        if len(cameras) < 4 and preselection != "exaustive": 
            logger.warning("Less than 4 cameras availabe. Performing exaustive preseleection")
        
         
    def matching(self, cfg: dict, images: List[ImageDS], epoch_dict:dict, epoch: int):
        features = MatchingAndTracking(
            cfg=cfg,
            epoch=epoch,
            images=images,
            features=features,
            epoch_dict=epoch_dict,
        )
    
    