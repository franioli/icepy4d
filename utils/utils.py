"""

"""

from pathlib import Path
import argparse
import random
import numpy as np
# import  pydegensac
from copy import deepcopy
import cv2 
import matplotlib.pyplot as plt
import os, glob


class Camera:
    """ Class to help manage Cameras. """
    
    """Initialise the camera calibration object 
    Parameters
    ----------
    *args : str
      Read calibration text file, with Full OpenCV parameters in a row.
    """

    def __init__(self, *args):
        
        self.reset()
        if args[0] == None:
            print('No calibration file available. Setting default parameters')
        self.K = np.array([[6620.56653699, 0., 3020.0385], [0., 6619.88882, 1886.01352], [0., 0., 1.]] )
        
    
    def reset(self):
        self.K = np.array([[500, 0., 500], [0., 500, 500], [0., 0., 1.]] )
        self.R = np.identity(3)
        self.t = np.zeros((3,))
        self.P = cv2.projectionFromKRt(self.K, self.R, self.t)
        
               
class images:
    """ Class to help manage Cameras. """
    
    """Initialise the camera calibration object 
    Parameters
    ----------
    *args : str
      Read calibration text file, with Full OpenCV parameters in a row.
    """

    def __init__(self, *args):
        
        self.reset()

        
    def reset(self):
        
        
               