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
import  pydegensac
# from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.cm as cm
# import json

from utils.utils import read_img
from utils.match_pairs import match_pair
from utils.track_matches import track_matches
from utils.sg.utils import make_matching_plot

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

# epoch = 0
# if epoch == 0:
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
    matchedPts, matchedDescriptors, matchedPtsScores, _ = match_pair(pair, maskBB, opt_matching)
    
    if epoch == 0:
        # Store matches in features structure
        features = [{   'mkpts0': matchedPts['mkpts0'], 
                        'mkpts1': matchedPts['mkpts1'],
                        # 'mconf': matchedPts['match_confidence'],
                        'descr0': matchedDescriptors[0], 
                        'descr1': matchedDescriptors[1],
                        'scores0': matchedPtsScores[0], 
                        'scores1': matchedPtsScores[1] }] 
    
    
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
                           }   
        
        prevPts_cam0 = {'keypoints0': features[epoch-1]['mkpts0'], 
                       'descriptors0': features[epoch-1]['descr0'],
                       'scores0': features[epoch-1]['scores0']}
        
        prevPts_cam1 = {'keypoints0': features[epoch-1]['mkpts1'], 
                       'descriptors0': features[epoch-1]['descr1'], 
                       'scores0': features[epoch-1]['scores1']}
        tracked_cam0, tracked_cam1 = track_matches(pairs, maskBB, [prevPts_cam0, prevPts_cam1], opt_tracking)
        # TO DO: tenere traccia anche dei descriptors and scores dei punti traccati!
        
        # Store all matches in features structure
        features.append({'mkpts0': np.concatenate((matchedPts['mkpts0'], tracked_cam0['keypoints1']), axis=0 ), 
                         'mkpts1': np.concatenate((matchedPts['mkpts1'], tracked_cam1['keypoints1']), axis=0 ),
                         # 'mconf': matchedPts['match_confidence'],
                          'descr0': np.concatenate((matchedDescriptors[0], tracked_cam0['descriptors1']), axis=1 ),
                          'descr1': np.concatenate((matchedDescriptors[1], tracked_cam1['descriptors1']), axis=1 ),
                          'scores0': np.concatenate((matchedPtsScores[0], tracked_cam0['scores1']), axis=0 ), 
                          'scores1': np.concatenate((matchedPtsScores[1], tracked_cam1['scores1']), axis=0 ), 
                         })
                         
        F, inlMask= pydegensac.findFundamentalMatrix(features[epoch]['mkpts0'], features[epoch]['mkpts1'], px_th=3, conf=0.9,
                                                      max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson',
                                                      symmetric_error_check=True, enable_degeneracy_check=True)
        print('Matches at epoch {}: pydegensac found {} inliers ({:.2f}%)'.format(epoch, int(deepcopy(inlMask).astype(np.float32).sum()),
                        int(deepcopy(inlMask).astype(np.float32).sum())*100 / len(features[epoch]['mkpts0'])))
        
        
        # # Visualize the matches.
        # image0 = read_img(images[0][epoch], 0, [-1], [])[0]
        # image1 = read_img(images[1][epoch], 0, [-1], [])[0]
        # mkpts0 = features[epoch]['mkpts0']
        # mkpts1 = features[epoch]['mkpts1']
        # color= cm.jet( range(1, len(features[epoch]['mkpts0'])) )
        # text= ['SuperGlue', 'Matches: {}'.format(len(features[epoch]['mkpts0']))]
        # small_text= ['Image Pair: {}:{}'.format(images[0][epoch], images[1][epoch])]
        # viz_matches_epoch(
        #     image0, image1, features[epoch]['mkpts0'], features[epoch]['mkpts1'],
        #     features[epoch]['mkpts0'], features[epoch]['mkpts1'],
        #     color, text, epochdir, False, False, False, 'Matches', small_text)



# def viz_matches_epoch(image0, image1, kpts0, kpts1, mkpts0,
#                             mkpts1, color, text, path=None,
#                             show_keypoints=False, margin=10,
#                             opencv_display=False, opencv_title='',
#                             small_text=[]):
#     H0, W0 = image0.shape
#     H1, W1 = image1.shape
#     H, W = max(H0, H1), W0 + W1 + margin

#     out = 255*np.ones((H, W), np.uint8)
#     out[:H0, :W0] = image0
#     out[:H1, W0+margin:] = image1
#     out = np.stack([out]*3, -1)

#     kpts0, kpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
#     mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)

#     white = (255, 255, 255)
#     black = (0, 0, 0)
#     for x, y in kpts0:
#         cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
#         cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
#     for x, y in kpts1:
#         cv2.circle(out, (x + margin + W0, y), 2, black, -1,
#                     lineType=cv2.LINE_AA)
#         cv2.circle(out, (x + margin + W0, y), 1, white, -1,
#                     lineType=cv2.LINE_AA)

#     color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
#     for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
#         c = c.tolist()
#         cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
#                  color=c, thickness=1, lineType=cv2.LINE_AA)
#         # display line end-points as circles
#         cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
#         cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
#                    lineType=cv2.LINE_AA)

#     # Scale factor for consistent visualization across scales.
#     sc = min(H / 640., 2.0)

#     # Big text.
#     Ht = int(30 * sc)  # text height
#     txt_color_fg = (255, 255, 255)
#     txt_color_bg = (0, 0, 0)
#     for i, t in enumerate(text):
#         cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
#                     1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
#         cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
#                     1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

#     # Small text.
#     Ht = int(18 * sc)  # text height
#     for i, t in enumerate(reversed(small_text)):
#         cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
#                     0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
#         cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
#                     0.5*sc, txt_color_fg, 1, cv2.LINE_AA)
#     cv2.imwrite(os.path.join(epochdir,'_matches.png'), out)
    
#     return out




#  drawMatches(
#      image0, features[epoch]['mkpts0'], image1, features[epoch]['mkpts1'],
#      list(range(1, len(features[epoch]['mkpts0']))) )

#     out_img = np.array([])
#     kp1 = cv2.KeyPoint.convert(features[epoch]['mkpts0'])
#     kp2 =  cv2.KeyPoint.convert(features[epoch]['mkpts1'])
    
#     matches = [i for i in range(1, len(features[epoch]['mkpts0'])+1) ]
    
#     matches = list(range(1, len(features[epoch]['mkpts0'])+1))
# 	cv2.drawMatches(img1, kp1, img2, kp2, matches, out_img)
                
                
# def drawMatches(img1, kp1, img2, kp2, matches):
#     """
#     My own implementation of cv2.drawMatches as OpenCV 2.4.9
#     does not have this function available but it's supported in
#     OpenCV 3.0.0

#     This function takes in two images with their associated 
#     keypoints, as well as a list of DMatch data structure (matches) 
#     that contains which keypoints matched in which images.

#     An image will be produced where a montage is shown with
#     the first image followed by the second image beside it.

#     Keypoints are delineated with circles, while lines are connected
#     between matching keypoints.

#     img1,img2 - Grayscale images
#     kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
#               detection algorithms
#     matches - A list of matches of corresponding keypoints through any
#               OpenCV keypoint matching algorithm
#     """

#     # Create a new output image that concatenates the two images together
#     # (a.k.a) a montage
#     rows1 = img1.shape[0]
#     cols1 = img1.shape[1]
#     rows2 = img2.shape[0]
#     cols2 = img2.shape[1]

#     out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

#     # Place the first image to the left
#     out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

#     # Place the next image to the right of it
#     out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

#     # For each pair of points we have between both images
#     # draw circles, then connect a line between them
#     for mat in matches:

#         # Get the matching keypoints for each of the images
#         img1_idx = mat.queryIdx
#         img2_idx = mat.trainIdx

#         # x - columns
#         # y - rows
#         (x1,y1) = kp1[img1_idx].pt
#         (x2,y2) = kp2[img2_idx].pt

#         # Draw a small circle at both co-ordinates
#         # radius 4
#         # colour blue
#         # thickness = 1
#         cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
#         cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

#         # Draw a line in between the two points
#         # thickness = 1
#         # colour blue
#         cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


#     # Show the image
#     cv2.imshow('Matched Features', out)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()







