'''
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
'''
import os
import cv2 
import pickle
import numpy as np

from scipy import linalg
from pathlib import Path

from lib.geometry import (P_from_KRT, 
                          # C_from_P
                          )
# from geometry import (P_from_KRT, C_from_P)

#--- Camera ---#
class Camera:
    ''' Class to help manage Cameras. '''

    def __init__(self, K=None, R=None, t=None, dist=None, calib_path=None):
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
        #TODO: add assertion to check that only K and dist OR calib_path is provided.
             
        self.K = K # calibration matrix
        self.dist = dist # Distortion vector in OpenCV format
        self.R = R # rotation
        self.t = t # translation
        self.P = None
        self.C = None # camera center
        self.pose = None
        self.extrinsics = None
        
        # If calib_path is provided, read camera calibration from file 
        if calib_path is not None:
            self.read_calibration_from_file(calib_path)
        
        if R is None and t is None: 
            self.reset_EO()
            # self.compose_P()
            # self.C_from_P()
            
    def reset_EO(self):
        ''' Reset camera EO as to make camera reference system parallel to world reference system '''
        self.extrinsics = np.eye(4)
        self.update_camera_from_extrinsics()
        self.extrinsics_to_pose()
        self.C_from_P()
        # self.R = np.identity(3)
        # self.t = np.zeros((3,)).reshape(3,1)
        # self.P = P_from_KRT(self.K, self.R, self.t)
        # self.C_from_P()
      
    def Rt_to_extrinsics(self):        
        ''' 
        [ R | t ]    [ I | t ]   [ R | 0 ]
        | --|-- |  = | --|-- | * | --|-- |  
        [ 0 | 1 ]    [ 0 | 1 ]   [ 0 | 1 ]
        '''       
        # t = np.block([[np.eye(3), self.t], 
        #               [np.zeros((1,3)), 1]]
        #              )
        # R = np.block([[self.R, np.zeros((3,1))],
        #               [np.zeros((1,3)), 1]]
        #              )
        R_block = self.build_block_matrix(self.R)
        t_block = self.build_block_matrix(self.t)
        
        self.extrinsics = np.dot(t_block, R_block)
        
        return self.extrinsics
        
    def extrinsics_to_pose(self):        
        ''' 
        '''
        if self.extrinsics is None:
            self.Rt_to_extrinsics()
        
        R = self.extrinsics[0:3,0:3]
        t = self.extrinsics[0:3,3:4]
        
        Rc = R.T
        C = -np.dot(Rc, t)       
        
        Rc_block = self.build_block_matrix(Rc)
        C_block = self.build_block_matrix(C)

        self.pose = np.dot(C_block, Rc_block)

        return self.pose
        
    def pose_to_extrinsics(self):        
        ''' 
       
        '''
        if self.pose is None:
            print('Camera pose not available. Compute it first.' )
            return None
        else: 
            Rc = self.pose[0:3,0:3]
            C = self.pose[0:3,3:4]
            
            R = Rc.T
            t = -np.dot(R, C)
            
            t_block = self.build_block_matrix(t)
            R_block = self.build_block_matrix(R)
            self.extrinsics = np.dot(t_block, R_block)
            self.update_camera_from_extrinsics()
            
            return self.extrinsics
    
    def update_camera_from_extrinsics(self):        
        ''' 
       
        '''
        if self.extrinsics is None:
            print('Camera extrinsics not available. Compute it first.' )
            return None
        else: 
            self.R = self.extrinsics[0:3,0:3]
            self.t = self.extrinsics[0:3,3:4]
            self.P = np.dot(self.K, self.extrinsics[0:3,:])
            
    def get_C_from_pose(self):        
         ''' 

         '''
         return self.pose[0:3,3:4]
            
    def C_from_P(self):        
        ''' 
        Compute and return the camera center from projection matrix P, as
        C = [ - inv(KR) * Kt ] = [ -inv(P[1:3]) * P[4] ]
        '''
        # if self.C is not None:
        #     return self.C
        # else:
        self.C = -np.dot(np.linalg.inv(self.P[:,0:3]), self.P[:,3].reshape(3,1) )
        return self.C
        
    def t_from_RC(self):        
        ''' Deprecrated function. Use extrinsics_to_pose instead.
        Compute and return the camera translation vector t, given the camera 
        centre and the roation matrix X, as
        t = [ -R * C ] 
        The relation is derived from the formula of the camera centre
        C = [ - inv(KR) * Kt ]
        '''
        self.t = -np.dot(self.R, self.C)
        self.compose_P()
        return self.t        
    
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
        ''' Factorize the camera matrix into K,R,t as P = K[R|t]. '''
        
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
       
    def read_calibration_from_file(self, path):
        '''
        Read camera internal orientation from file, save in camera class
        and return them.
        The file must contain the full K matrix and distortion vector, 
        according to OpenCV standards, and organized in one line, as follow:
        fx 0. cx 0. fy cy 0. 0. 1. k1, k2, p1, p2, [k3, [k4, k5, k6
        Values must be float (include the . after integers) and divided by a 
        white space. 
        -------      
        Returns:  K, dist
        '''
        path = Path(path)
        if not path.exists():
            print('Error: calibration filed does not exist.')
            return None, None       
        with open(path, 'r') as f:
            data = np.loadtxt(f)
            K = data[0:9].astype(float).reshape(3, 3, order='C')
            if len(data) == 13:
                print('Using OPENCV camera model.')
                dist = data[9:13].astype(float)
            elif len(data) == 14:
                print('Using OPENCV camera model + k3')
                dist = data[9:14].astype(float)
            elif len(data) == 17:
                print('Using FULL OPENCV camera model')
                dist = data[9:17].astype(float)    
            else:
                print('invalid intrinsics data.')
                return None, None
            # TODO: implement other camera models and estimate K from exif.
        self.K = K
        self.dist = dist
        return K, dist

    def euler_from_R(self):
        '''
        Compute Euler angles from rotation matrix
        -------      
        Returns:  [omega, phi, kappa]
        '''
        omega = np.arctan2(self.R[2,1], self.R[2,2]) 
        phi = np.arctan2(-self.R[2,0], np.sqrt(self.R[2,1]**2+self.R[2,2]**2)); 
        kappa = np.arctan2(self.R[1,0], self.R[0,0]); 
        
        return [omega, phi, kappa]
    

    def build_block_matrix(self, mat):
        # TODO: add description
        ''' 
    
        '''
        if mat.shape[1] == 3:
            block = np.block([[mat, np.zeros((3,1))], 
                              [np.zeros((1,3)), 1]]
                              )
        elif mat.shape[1] == 1:
            block = np.block([[np.eye(3), mat], 
                              [np.zeros((1,3)), 1]]
                              )     
        else:
            print('Error: unknown input matrix dimensions.')
            return None
            
        return block

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
        #TODO: change name in read image list
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
  
    
# Targets  
class Targets:
    ''' 
    Class to store Target information, including image coordinates and object coordinates
    Targets are stored as numpy arrays: 
        Targets.im_coor: [nx2] List of array of containing xy coordinates of the 
                        target projections on each image
        Targets.obj_coor: nx3 array of XYZ object coordinates (it can be empty)
    '''
    def __init__(self, cam_id=None, im_coord_path=None):
        self.reset_targets()
        
        # If cam_id and im_coord_path are rpovided, read image coordinates from file
        if im_coord_path is not None and cam_id is not None:
            if type(cam_id) == list and type(im_coord_path) == list:
                if len(cam_id) != len(im_coord_path):
                    print('Error: diffent number of elements in cameras id \
                          and paths provided.')
                    return
                for cam, path in zip(cam_id, im_coord_path):
                    self.read_im_coord_from_txt(cam, path)
            else:
                self.read_im_coord_from_txt(cam_id, im_coord_path)
        
    def __len__(self):
        ''' Get total number of featues stored'''
        return len(self.im_coor)        
        
    def reset_targets(self):
        '''
        Reset Target instance to empy list and None objects
        '''
        self.im_coor = []
        self.obj_coor = None
    
    def get_im_coord(self, cam_id=None, epoch=None):
        ''' 
        Return image coordinates as numpy array 
        If numeric camera id (integer) is provided, the function returns the
        image coordinates in that camera, otherwise the list with the projections
        on all the cameras is returned.
        '''
        if cam_id is None:
            return np.float32(self.im_coor)
        else:
            if epoch is None:
                return np.float32(self.im_coor[cam_id])
            elif epoch < len(self):
                return np.float32(self.im_coor[cam_id][epoch])
    
    def get_obj_coord(self):
        ''' Return objject coordinates as numpy array '''
        return np.float32(self.obj_coor)
    
    def append_features(self, new_features):
        print('method not implemented yet')
        
    def append_obj_cord(self, new_obj_coor):
        #TODO: add check on dimension and add description
        if self.obj_coor is None:
            self.obj_coor = new_obj_coor
        else:
            self.obj_coor = np.append(self.obj_coor, new_obj_coor, axis=0)
            
    def read_im_coord_from_txt(self, camera_id=None, path=None, fmt='%i', delimiter=',', header='x,y'):
        ''' 
        Read image target image coordinates from .txt file, organized as follows:
            - One line per target
            - first x coordinate, then y coordinate
            - Coordinates separated by a delimiter (default ',')          
            e.g. 
            #x,y
            1000,2000
            2000,3000
            
            NB: added -1 in the image coordinates to take into account 
            matlab-python different image coordinates
        '''
        if camera_id is None:
            print('Error: missing camera id. Impossible to assign the target\
                  coordinates to the correct camera')
            return
        if path is None:
            print("Error: missing path argument.")
            return
        path = Path(path)
        if not path.exists():
            print('Error: Input path does not exist.') 
            return
        with open(path, 'r') as f:
            data = np.loadtxt(f, delimiter=',' )
            data = data - 1
        self.im_coor.insert(camera_id,data)
        
    def save_as_txt(self, path=None, fmt='%i', delimiter=',', header='x,y'):
        ''' Save keypoints in a .txt file '''
        if path is None:
            print("Error: missing path argument.")
            return
        # if not Path(path).:
        #     print('Error: invalid input path.')
        #     return
        np.savetxt(path, self.kpts, fmt=fmt, delimiter=delimiter, newline='\n', header=header) 
      
# #--- DSM ---#  
# class DSM:
#     ''' Class to store and manage DSM. '''
#     def __init__(self, x, y, z, res):
#         xx, yy = np.meshgrid(x,y)
#         self.x = xx
#         self.y = yy
#         self.z = z
#         self.res = res    
                    
 
        
if __name__ == '__main__':
    '''Test classes '''
    
