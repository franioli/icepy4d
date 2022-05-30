# Belvedere stereo matching
# 
# 
# 
# 
# 
# v0.1 2022.05.17


from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import  pydegensac
from copy import deepcopy
import cv2 

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
    
    
    imFld           = 'data/IMG'
    imExt           = '.JPG'
    calibFld        = 'data/calib'
    
    numCams         = 2
     
    
    undistimgs      = True
    enhanceimgs     = True
    printFigs       = False
    useTiles        = False
    
    warpImages      = False
    
#%% Initialize structures for storing data
    cameras         = {}                                                            # Dict for storing cameras information
    images          = {'imds': [], 'exif': []}                                      # Dict for storing image datastore strctures
    im              = []                                                            # List for storing image pairs
    
    matchedPts      = {}                                                            # Dict for storing double points found at each epoch by DFM or SuperGlue
    trackedPoints   = {}                                                            # Dict for storing tracked points for each camera in time
    features        = {}                                                            # Dict for storing all the valid matched features at all epochs
    sparsePts       = {'labels': [], 'pointsXYZ': []}                               # Dict for storing point clouds at all epochs
    
    fMats = []
    
    maskBB          = []
    maskGlacier     = []
    
#%% Load data
    
    print('Loading data:...')
    
    d = os.listdir(os.path.join(rootDirPath, imFld))
    
    
    %- Get lists of images and calibration data
    d = dir(fullfile(rootDirPath, imFld)) ;
    % d = d([d(:).isdir] & ~ismember({d(:).name},{'.','..'}));
    % d = d(ismember({d(:).name},{'p1','p2'}));
    d = d(ismember({d(:).name},{'p2','p3'}));
    c = dir(fullfile(rootDirPath, calibFld,'*.mat'));                           % Rivedere assegnazione file calibrazione a camera.
    
    %- inizialize camera structures
    for jj = 1:numCams
        a = load(fullfile(rootDirPath, calibFld, c(jj).name));      
        cameras(jj).cameraParams   = a.cameraParams;  
        cameras(jj).K       = cameras(jj).cameraParams.IntrinsicMatrix';
        cameras(jj).t{1}    = zeros(3,1);     
        cameras(jj).R{1}    = eye(3,3);
        cameras(jj).P{1}    = cameras(jj).K * [cameras(jj).R{1} cameras(jj).t{1}];
        cameras(jj).X0{1}   = - cameras(jj).P{1}(:,1:3)\cameras(jj).P{1}(:,4);
    end
    
    %- inizialize image datastore structures
    warning off
    for jj = 1:numCams
        images(jj).imds    = imageDatastore(fullfile(rootDirPath, imFld, d(jj).name),'FileExtensions',{imExt}, 'LabelSource','foldernames'); 
        for imgId = 1:length(images(jj).imds.Files)
            images(jj).exif    = imfinfo(images(jj).imds.Files{imgId});
        end
    end
    clearvars a c d jj
    warning on      
    
    %- Load mask for cropping images  (To do: allow also polygon mask)
    if exist(fullfile(rootDirPath, imFld,'mask.mat'), 'file')
        load(fullfile(rootDirPath, imFld,'mask.mat'))
    else
        maskBB = struct('roi', []);
        for jj = 1:numCams
            mskFig = figure('Name', 'Mask');     
            im{jj} = readimage(images(jj).imds,1); 
            ax = mskFig.CurrentAxes; 
            imshow(im{jj});  
            roi = drawrectangle(ax);
            maskBB(jj).roi = roi;
            maskBB(jj).pos = round(maskBB(jj).roi.Position/100)*100;
        end
        save(fullfile(rootDirPath, imFld,'mask.mat'), 'maskBB')    
        close(mskFig), clearvars ax fig roi
    end
    
    %- load mask on glacier
    if exist(fullfile(rootDirPath, imFld,'maskGlacier.mat'), 'file')
        load(fullfile(rootDirPath, imFld,'maskGlacier.mat'))
    else
        maskGlacier = struct('roi', []);
        for jj = 1:numCams
            mskFig = figure('Name', 'Mask');     
            im{jj} = readimage(images(jj).imds,1); 
            ax = mskFig.CurrentAxes; 
            imshow(im{jj});  
            roi = drawpolygon(ax);
            maskGlacier(jj).roi = roi;
        end
        save(fullfile(rootDirPath, imFld,'maskGlacier.mat'), 'maskGlacier')    
        close(mskFig), clearvars ax fig roi I1
    end
    
    %- load DFM network
    % dfmModel = load(fullfile(rootDirPath, 'mat', 'thirdParts', 'DFM\models\imagenet-vgg-verydeep-19.mat'));
    
    fprintf('Done.\n')
