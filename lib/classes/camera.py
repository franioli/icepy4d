import numpy as np
from scipy import linalg
import cv2 

from lib.geometry import (P_from_KRT, X0_from_P)

class Camera:
    """ Class to help manage Cameras. """

    def __init__(self, K=None, R=None, t=None, dist=None):
        """ Initialize pinhole camera model """
        #TODO: add checks on inputs
        # If not None, convert inputs to np array
        if K is not None:
            K = np.array(K)
        if R is not None:
            R = np.array(R)
        if t is not None:
            t = np.array(t)       
        if dist is not None:
            dist = np.array(dist)                
        self.K = K # calibration matrix
        self.R = R # rotation
        self.t = t # translation
        self.P = None
        self.X0 = None # camera center
        self.dist = dist # Distortion vector in OpenCV format
  
    def reset_EO(self):
        ''' Reset camera EO as to make camera reference system parallel to world reference system'''
        self.R = np.identity(3)
        self.t = np.zeros((3,)).reshape(3,1)
        self.P = P_from_KRT(self.K, self.R, self.t)
        self.X0 = X0_from_P(self.P)
        
    def camera_center(self):        
        """ Compute and return the camera center. """
        if self.X0 is not None:
            return self.X0
        else:
            self.X0 = -np.dot(self.R.T,self.t)
            return self.X0
    
    def compose_P(self):
        """
        Compose and return the 4x3 P matrix from 3x3 K matrix, 3x3 R matrix and 3x1 t vector, as:
            K[ R | t ]
        """
        if (self.K is None):
            print("Invalid calibration matrix. Unable to compute P.")
            self.P = None 
            return None
        elif (self.R is None):
            print("Invalid Rotation matrix. Unable to compute P.")
            self.P = None 
            return None
        elif (self.t is None):
            print("Invalid translation vector. Unable to compute P.")
            self.P = None 
            return None
                         
        RT = np.zeros((3,4))
        RT[:, 0:3] = self.R
        RT[:, 3:4] = self.t
        self.P = np.dot(self.K,RT) 
        return self.P
        
    def factor_P(self):
        """  Factorize the camera matrix into K,R,t as P = K[R|t]. """
        
        # factor first 3*3 part
        K,R = linalg.rq(self.P[:,:3])
        
        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1
        
        self.K = np.dot(K,T)
        self.R = np.dot(T,R) # T is its own inverse
        self.t = np.dot(linalg.inv(self.K),self.P[:,3]).reshape(3,1)
        
        return self.K, self.R, self.t
    
    

if __name__ == '__main__':
   
    K = [[6900.766178626993, 0.0, 3055.9219427396583], [0.0, 6919.0517432373235, 1659.8768050681379], [0.0, 0.0, 1.0]]
    dist = [-0.07241143420209739, 0.00311945599198001, -0.008597066196675609, 0.002601995972163532, 0.46863386164346776]
    cam = Camera(K=K, dist=dist)
   

        
# class Camera:
#     """ Class to help manage Cameras. """
    
#     """Initialise the camera calibration object 
#     Parameters
#     ----------
#     *args : str
#       Read calibration text file, with Full OpenCV parameters in a row.
#     """[CVPR 2022] Learning Graph Regularisation for Guided Super-Resolution 

#     def __init__(self, *args):
        
#         self.reset()
#         if args[0] == None:
#             print('No calibration file available. Setting default parameters')
#         self.K = np.array([[6620.56653699, 0., 3020.0385], [0., 6619.88882, 1886.01352], [0., 0., 1.]] )
        
    
#     def reset(self):
#         self.K = np.array([[500, 0., 500], [0., 500, 500], [0., 0., 1.]] )
#         self.R = np.identity(3)
#         self.t = np.zeros((3,))
#         self.P = cv2.projectionFromKRt(self.K, self.R, self.t)
        
               
# class images:
#     """ Class to help manage Cameras. """
    
#     """Initialise the camera calibration object 
#     Parameters
#     ----------
#     *args : str
#       Read calibration text file, with Full OpenCV parameters in a row.
#     """

#     def __init__(self, *args):
        
#         self.reset()

        
#     def reset(self):