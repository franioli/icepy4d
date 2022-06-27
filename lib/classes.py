import numpy as np
from scipy import linalg
import cv2 
import os
from pathlib import Path
import pickle

from lib.geometry import (P_from_KRT, X0_from_P)
# from geometry import (P_from_KRT, X0_from_P)

#--- Camera ---#
class Camera:
    ''' Class to help manage Cameras. '''
    
    # def __init__(self):
    #     '''Initialise the camera calibration object '''
    # Parameters
    # ----------
    # *args : str
    #   Read calibration text file, with Full OpenCV parameters in a row.
    #     print('Class not defined yet...')
    # TODO: implement method for reading calibration data from file

    def __init__(self, K=None, R=None, t=None, dist=None):
        ''' Initialize pinhole camera model '''
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
        if R is None and t is None: 
            self.reset_EO()
            self.compose_P()
            self.camera_center()
            
    def reset_EO(self):
        ''' Reset camera EO as to make camera reference system parallel to world reference system '''
        self.R = np.identity(3)
        self.t = np.zeros((3,)).reshape(3,1)
        self.P = P_from_KRT(self.K, self.R, self.t)
        self.X0 = X0_from_P(self.P)
        
    def camera_center(self):        
        ''' Compute and return the camera center. '''
        # if self.X0 is not None:
        #     return self.X0
        # else:
        self.X0 = -np.dot(self.R.T,self.t)
        return self.X0
    
    def compose_P(self):
        '''
        Compose and return the 4x3 P matrix from 3x3 K matrix, 3x3 R matrix and 3x1 t vector, as:
            K[ R | t ]
        '''
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
        '''  Factorize the camera matrix into K,R,t as P = K[R|t]. '''
        
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
    
    def project_points(self, points3d):
        '''
        Overhelmed method (see lib.geometry) for projecting 3D to image coordinates.
        
        Project 3D points (Nx3 array) to image coordinates, given the projection matrix P (4x3 matrix)
        If K matric and dist vector are given, the function computes undistorted image projections (otherwise, zero distortions are assumed)
        Returns: 2D projected points (Nx2 array) in image coordinates
        '''
        points3d = cv2.convertPointsToHomogeneous(points3d)[:,0,:]
        m = np.dot(self.P, points3d.T)
        m = m[0:2,:] / m[2,:]; 
        m = m.astype(float).T
        
        if self.dist is not None and self.K is not None:
            m = cv2.undistortPoints(m, self.K, self.dist, None, self.K)[:,0,:]
            
        return m.astype(float)


#--- Images ---#
class Imageds:
    '''
    Class to help manage Image datasets 
    
    '''
    def __init__(self, path=None):
        #TODO: implement labels in datastore
        if not hasattr(self, 'files'):          
            self.reset_imageds()
        if path is not None:
            self.get_image_list(path)

    def __len__(self):
        ''' Get number of images in the datastore '''
        return len(self.files)
        
    def __contains__(self, name):
        ''' Check if an image is in the datastore, given the image name'''
        return name in self.files
  
    def __getitem__(self, idx, **args):
        ''' Read and return the image at position idx in the image datastore '''
        # TODO: add possibility to chose reading between col or grayscale, scale image, crop etc...
        img = read_img2(os.path.join(self.folder[idx], self.files[idx]))
        if img is not None:
            print(f'Loaded image {self.files[idx]}')
        return img
    
    def reset_imageds(self):
        ''' Initialize image datastore '''
        self.files = []
        self.folder = []
        self.ext = []
        self.label = []
        # self.size = []
        # self.shot_date = []
        # self.shot_time = []
    
    def get_image_list(self, path):
        # TODO: add option for including subfolders
        if not os.path.exists(path):
            print('Error: invalid input path.')
            return
        d = os.listdir(path)
        d.sort()
        self.files = d
        self.folder = [path] * len(d)
    
    def get_image_name(self, idx):
        ''' Return image name at position idx in datastore '''
        return self.files[idx]
    
    def get_image_path(self, idx):
        ''' Return full path of the image at position idx in datastore '''
        return (os.path.join(self.folder[idx], self.files[idx]))
    
    def get_image_stem(self, idx):
        ''' Return name without extension (stem) of the image at position idx in datastore '''
        return Path(self.files[idx]).stem
    
    # TODO: Define iterable
    # def __iter__(self):
    # def __next__(self):
        
    # def __getattr__(self, name):
    #     return self._data[name]
    
#TODO: move to io.py lib and check all input/output variables when fct is called!
def read_img2(path, color=True, resize=[-1], crop=None):
    '''
    '''
    if color:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)
    
    if image is None:
        if len(resize) == 1 and resize[0] == -1:
            return None
        else:
            return None, None
    
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    image = cv2.resize(image, (w_new, h_new))
    if crop:
        image = image[ crop[1]:crop[3],crop[0]:crop[2] ]

    if len(resize) == 1 and resize[0] == -1 :
        return image
    else:
        return image, scales

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]
    return w_new, h_new
     
  
    
#--- Features ---#
class Features:
    ''' 
    Class to store matched features, descriptors and scores 
    Features are stored as numpy arrays: 
        Features.kpts: nx2 array of features location
        Features.descr: mxn array of descriptors (note that descriptors are stored columnwise)
        Features.score: nx1 array with feature score    '''

    def __init__(self):
        self.reset_fetures()
        
    def __len__(self):
        ''' Get total number of featues stored'''
        return len(self.kpts)        
        
    def reset_fetures(self):
        '''
        Reset Feature instance to None Objects
        '''
        self.kpts = None
        self.descr = None
        self.score = None        
    
    def initialize_fetures(self, nfeatures=1, descr_size=256):
        '''
        Inizialize Feature instance to numpy arrays, 
        optionally for a given number of features and descriptor size (default is 256).
        '''
        self.kpts = np.empty((nfeatures,2), dtype=float)
        self.descr = np.empty((descr_size,nfeatures), dtype=float)
        self.score = np.empty(nfeatures, dtype=float)
        
    def get_keypoints(self):
        ''' Return keypoints as numpy array '''
        return np.float32(self.kpts)
    
    def get_descriptors(self):
        ''' Return descriptors as numpy array '''
        return np.float32(self.descr)
    
    def get_scores(self):
        ''' Return scores as numpy array '''
        return np.float32(self.score)
    
    def get_features_as_dict(self):
        ''' Return a dictionary with keypoints, descriptors and scores, organized for SuperGlue'''
        out = { 'keypoints0': self.get_keypoints(), 
                'descriptors0': self.get_descriptors(),
                'scores0': self.get_scores() }
        return out
    
    def append_features(self, new_features):
        '''
        Append new features to Features Class. 
        Input new_features is a Dict with keys as follows:
            new_features['kpts']: nx2 array of features location
            new_features['descr']: mxn array of descriptors (note that descriptors are stored columnwise)
            new_features['score']: nx1 array with feature score
        '''
        # Check dictionary keys: 
        keys = ['kpts', 'descr', 'score']
        if any(key not in new_features.keys() for key in keys):
            print('Invalid input dictionary. Check all keys ["kpts", "descr", "scores"] are present')
            return self
        # TODO: check correct shape of inputs.
            
        if self.kpts is None :
            self.kpts = new_features['kpts']
            self.descr = new_features['descr']
            self.score = new_features['score']
        else: 
            self.kpts = np.append(self.kpts, new_features['kpts'], axis=0)
            self.descr = np.append(self.descr, new_features['descr'], axis=1)
            self.score = np.append(self.score, new_features['score'], axis=0)
            
    def save_as_pickle(self, path=None):
        ''' Save keypoints in a .txt file '''
        if path is None:
            print("Error: missing path argument.")
            return
        # if not Path(path).:
        #     print('Error: invalid input path.')
        #     return
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)         
        
    def save_as_txt(self, path=None, fmt='%i', delimiter=',', header='x,y'):
        ''' Save keypoints in a .txt file '''
        if path is None:
            print("Error: missing path argument.")
            return
        # if not Path(path).:
        #     print('Error: invalid input path.')
        #     return
        np.savetxt(path, self.kpts, fmt=fmt, delimiter=delimiter, newline='\n', header=header) 
  
    
#--- DSM ---#  
class DSM:
    ''' Class to store and manage DSM. '''
    def __init__(self, x, y, z, res):
        xx, yy = np.meshgrid(x,y)
        self.x = xx
        self.y = yy
        self.z = z
        self.res = res    
        
    # def generate_tif(self, ):
            
 
        
if __name__ == '__main__':
    '''Test classes '''
    
    # feat0 = Features()
    # nfeatures = 2
    # new_features = {'kpts': np.empty((nfeatures,2), dtype=float), 
    #              'descr': np.empty((256,nfeatures), dtype=float),  
    #              'score': np.empty(nfeatures, dtype=float) }
    
    # feat0.append_features(new_features)
    # feat0.append_features(new_features)
    # print(feat0.get_keypoints())
  
    