# Belvedere stereo matching
# 
# 
# 
# v0.1 2022.05.17

import numpy as np
import os
import cv2 
# import torch
# import argparse
# import random
# import time
# import  pydegensac
# from pathlib import Path
# from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
# import matplotlib.cm as cm
# import json

# from utils.utils import read_img
# from utils.tiles import *

from utils.match_pairs import match_pair
from utils.track_matches import track_matches

# from models.matching import Matching
# from models.utils import (compute_pose_error, compute_epipolar_error,
#                           estimate_pose, make_matching_plot,
#                           error_colormap, AverageTimer, pose_auc, read_image,
#                           rotate_intrinsics, rotate_pose_inplane,
#                           scale_intrinsics, frame2tensor,
#                           vizTileRes)
# # from  models.tiles import (subdivideImage, subdivideImage2, 
#                            appendPred, applyMatchesOffset)

    
#%%  Parameters (to be put in parser)

if __name__ == '__main__':
    rootDirPath = '.'
    
    #- Folders and paths
    imFld           = 'data/img'
    imExt           = '.tif'
    calibFld        = 'data/calib'
   
    #- CAMERAS
    numCams         = 2
    camNames        = ['p2', 'p3']
    maskBB          = [[600,1900,5300, 3600], [800,1800,5500,3500]]             # Bounding box for processing the images from the two cameras
    # maskBB          = [[400,1700,5500, 3800], [600,1600,5700,3700]]             # Bounding box for processing the images from the two cameras

    #- ON-OFF switches
    # undistimgs      = True
    # enhanceimgs     = True
    # printFigs       = False
    # useTiles        = False
    # warpImages      = False
    
    
#%%  Load data
    print('Loading data:...')

    cameras         = []                                                            # List for storing cameras information (as dicts)
    # images          = {'imds': [], 'exif': []}                                    # Dict for storing image datastore strctures
    images          = []                                                            # List for storing image paths
    # im              = []                                                            # List for storing image pairs
    
    features        = []                                                            # Dict for storing all the valid matched features at all epochs
    # sparsePts       = []                               # Dict for storing point clouds at all epochs
    
    # fMats = []
        
    #- images
    for jj, cam in enumerate(camNames):
        d  = os.listdir(os.path.join(rootDirPath, imFld, cam))
        for i, f in enumerate(d):
            d[i] = os.path.join(rootDirPath, imFld, cam, f)
        d.sort()
        if jj > 0 and len(d) is not len(images[jj-1]):
            print('Error: different number of images per camera')
        else:
            images.insert(jj, d)
       
    #- Cameras structures
    # TO DO: implement camera class!
    for jj, cam in enumerate(camNames):
        path = (os.path.join(rootDirPath, calibFld, cam+'.txt'))
        with open(path, 'r') as f:
            data = np.loadtxt(f)
        K = data[0:9].astype(float).reshape(3, 3, order='C')
        dist = data[10:15].astype(float)
        cameras.insert(jj, {'K': K, 'dist': dist})
       
    # im = cv2.imread(images[0][0], 1)
    # cv2.imshow('aa', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # plt.show()
     
    # Remove some variables
    del d, data, K, dist, path, f, i, jj
    
#%% Process epoch 

epoches2process = [0,1,2,3] #1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# numEpoch2track  = 1;

# epoch = 2
# if epoch == 2:
for epoch in epoches2process:
    print(f'Processing epoch {epoch}...')

    #=== Find Matches at current epoch ===#
    print('Run Superglue to find matches at epoch {}'.format(epoch))    
    epochdir = os.path.join('res','epoch_'+str(epoch))      
    pair = [images[0][epoch], images[1][epoch]]
    maskBB = np.array(maskBB).astype('int')
    opt_matching ={'output_dir': epochdir, 
                   
                   'resize': [-1],
                   'resize_float': True,
                   'equalize_hist': False,
    
                   'nms_radius': 3 , 
                   'keypoint_threshold': 0.0001, 
                   'max_keypoints': 4096, 
                   
                   'superglue': 'outdoor',
                   'sinkhorn_iterations': 100,
                   'match_threshold': 0.2, 
                 
                   'viz':  True,
                   'viz_extension': 'png',   # choices=['png', 'pdf'],
                   'fast_viz': True,
                   'opencv_display' : False, 
                   'show_keypoints': False, 
                                      
                   'cache': False,
                   'force_cpu': False,
                             
                   'useTile': True, 
                   'writeTile2Disk': False,
                   'do_viz_tile': False,
                   'rowDivisor': 2,
                   'colDivisor': 3,
                   'overlap': 300,            
                   }
    matchedPts, matchedPtsScores, matchedDescriptors, matchedTensors = match_pair(pair, maskBB, opt_matching)
    
    if epoch == 0:
        # Store matches in features structure
        features.insert(0, {'mkpts0': matchedPts['mkpts0'], 
                            'mkpts1': matchedPts['mkpts1'],
                            # 'mconf': matchedPts['match_confidence'],
                            # 'scores0': matchedPtsScores[0], 
                            # 'scores1': matchedPtsScores[1],
                            # 'descr0': matchedDescriptors[0], 
                            # 'descr1': matchedDescriptors[1] 
                            })
    
    
    #=== Track previous matches at current epoch ===#
    if epoch > 0:
        print('Track points from epoch {} to epoch {}'.format(epoch-1, epoch))
        trackoutdir = os.path.join('res','epoch_'+str(epoch), 'from_t'+str(epoch-1))
        pairs = [ [ images[0][epoch-1], images[0][epoch] ], 
                  [ images[1][epoch-1], images[1][epoch] ] ] 
        maskBB = np.array(maskBB).astype('int')
        opt_tracking = {'output_dir': trackoutdir,
                        
                        'resize': [-1],
                        'resize_float': True,
                        'equalize_hist': False,
         
                        'nms_radius': 3 , 
                        'keypoint_threshold': 0.0005, 
                        'max_keypoints': 8192, 
                        
                        'superglue': 'outdoor',
                        'sinkhorn_iterations': 100,
                        'match_threshold': 0.4, 
                      
                        'viz':  True,
                        'viz_extension': 'png',   # choices=['png', 'pdf'],
                        'fast_viz': True,
                        'opencv_display' : False, 
                        'show_keypoints': False, 
                        
                        'cache': False,
                        'force_cpu': False,
                                  
                        'useTile': True, 
                        'writeTile2Disk': False,
                        'do_viz_tile': False,
                        'rowDivisor': 2,
                        'colDivisor': 4,
                        'overlap': 0,            
                           }   
        trackedPoints = track_matches(pairs, maskBB, matchedTensors, opt_tracking)
        # TO DO: tenere traccia anche dei descriptors and scores dei punti traccati!
        
        # Store all matches in features structure
        features.append({'mkpts0': np.concatenate((matchedPts['mkpts0'], trackedPoints['mkpts0'].astype(float)), axis=0 ), 
                         'mkpts1': np.concatenate((matchedPts['mkpts1'], trackedPoints['mkpts1'].astype(float)), axis=0 ),
                         # 'mconf': matchedPts['match_confidence'],
                         # 'scores0': matchedPtsScores[0], 
                         # 'scores1': matchedPtsScores[1],
                         # 'descr0': matchedDescriptors[0], 
                         # 'descr1': matchedDescriptors[1] 
                         })
        
    # Write matched points to file
    np.savetxt( os.path.join(epochdir,'mkpts0_epoch'+str(epoch)+'.txt'), features[epoch]['mkpts0'],
               delimiter=',', fmt='%i')     
    np.savetxt( os.path.join(epochdir,'mkpts1_epoch'+str(epoch)+'.txt'), features[epoch]['mkpts1'],
               delimiter=',', fmt='%i')  


#%% Reconstruct geometry


