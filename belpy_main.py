import numpy as np
import os
from pathlib import Path
import cv2 
import pickle
import json 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d as o3d
import pydegensac

from lib.match_pairs import match_pair
from lib.track_matches import track_matches
# from src.sg.utils import make_matching_plot

from lib.io import read_img
from lib.geometry import (estimate_pose, P_from_KRT, X0_from_P, project_points)
from lib.utils import (normalize_and_und_points, draw_epip_lines, make_matching_plot, undistort_image, interpolate_point_colors, build_dsm)
from lib.thirdParts.triangulation import (linear_LS_triangulation, iterative_LS_triangulation)

#---  Parameters  ---#
# TODO: put parameters in parser or in json file

rootDirPath = '.'

#- Folders and paths
imFld = 'data/img'
imExt = '.tif'
calibFld = 'data/calib'
matching_config = 'config/opt_matching.json'
tracking_config = 'config/opt_tracking.json'

#- CAMERAS
numCams = 2
camNames = ['p2', 'p3']

#- Image cropping boundaries
# maskBB = [[600,1900,5300, 3600], [800,1800,5500,3500]]             # Bounding box for processing the images from the two cameras
maskBB = [[400,1500,5500,4000], [600,1400,5700,3900]]             # Bounding box for processing the images from the two cameras

#  Load data
cameras = []  # List for storing cameras information (as dicts)
images = []   # List for storing image paths
features = []  # Dict for storing all the valid matched features at all epochs
F_matrix = [] # List for storing fundamental matrixes
points3d = [] # List for storing 3D points

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
# TODO: change order of epoches and cameras to make everything consistent!
        
#- Cameras structures
# TO DO: implement camera class!
for jj, cam in enumerate(camNames):
    path = (os.path.join(rootDirPath, calibFld, cam+'.txt'))
    with open(path, 'r') as f:
        data = np.loadtxt(f)
    K = data[0:9].astype(float).reshape(3, 3, order='C')
    dist = data[9:13].astype(float)
    cameras.insert(jj, {'K': K, 'dist': dist})

# Remove some variables
del d, data, K, dist, path, f, i, jj

print('Data loaded')

#%% Perform matching and tracking
find_matches = 1
if find_matches:
    epoches2process = [0] # #1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    
    for epoch in epoches2process:
        print(f'Processing epoch {epoch}...')

        #=== Find Matches at current epoch ===#
        print('Run Superglue to find matches at epoch {}'.format(epoch))    
        epochdir = os.path.join('res','epoch_'+str(epoch))      
        with open(matching_config,) as f:
            opt_matching = json.load(f)
        opt_matching['output_dir'] = epochdir
        pair = [images[0][epoch], images[1][epoch]]
        maskBB = np.array(maskBB).astype('int')
        matchedPts, matchedDescriptors, matchedPtsScores = match_pair(pair, maskBB, opt_matching)

        # Store matches in features structure
        if epoch == 0:
            features = [{   'mkpts0': matchedPts['mkpts0'], 
                            'mkpts1': matchedPts['mkpts1'],
                            # 'mconf': matchedPts['match_confidence'],
                            'descr0': matchedDescriptors[0], 
                            'descr1': matchedDescriptors[1],
                            'scores0': matchedPtsScores[0], 
                            'scores1': matchedPtsScores[1] }] 
        # TODO: Store match confidence!

        #=== Track previous matches at current epoch ===#
        if epoch > 0:
            print('Track points from epoch {} to epoch {}'.format(epoch-1, epoch))
            
            trackoutdir = os.path.join('res','epoch_'+str(epoch), 'from_t'+str(epoch-1))
            with open(tracking_config,) as f:
                opt_tracking = json.load(f)
            opt_tracking['output_dir'] = trackoutdir
            pairs = [ [ images[0][epoch-1], images[0][epoch] ], 
                      [ images[1][epoch-1], images[1][epoch] ] ] 
            maskBB = np.array(maskBB).astype('int')
                            
            prevs = [{'keypoints0': np.float32(features[epoch-1]['mkpts0']), 
                      'descriptors0': np.float32(features[epoch-1]['descr0']),
                      'scores0': np.float32(features[epoch-1]['scores0']) }, 
                     {'keypoints0': np.float32(features[epoch-1]['mkpts1']), 
                      'descriptors0': np.float32(features[epoch-1]['descr1']), 
                      'scores0': np.float32(features[epoch-1]['scores1'])  }  ]
            tracked_cam0, tracked_cam1 = track_matches(pairs, maskBB, prevs, opt_tracking)
            # TO DO: tenere traccia anche dei descriptors and scores dei punti traccati!
            # TO DO: tenere traccia dell'epoca in cui è stato trovato il match
            # TO CHECK: Problema nei punti tracciati... vengono rigettati da pydegensac

            # Store all matches in features structure
            features.append({'mkpts0': np.concatenate((matchedPts['mkpts0'], tracked_cam0['keypoints1']), axis=0 ), 
                             'mkpts1': np.concatenate((matchedPts['mkpts1'], tracked_cam1['keypoints1']), axis=0 ),
                             # 'mconf': matchedPts['match_confidence'],
                              'descr0': np.concatenate((matchedDescriptors[0], tracked_cam0['descriptors1']), axis=1 ),
                              'descr1': np.concatenate((matchedDescriptors[1], tracked_cam1['descriptors1']), axis=1 ),
                              'scores0': np.concatenate((matchedPtsScores[0], tracked_cam0['scores1']), axis=0 ), 
                              'scores1': np.concatenate((matchedPtsScores[1], tracked_cam1['scores1']), axis=0 ), 
                             })

            # Run Pydegensac to estimate F matrix and reject outliers                         
            F, inlMask = pydegensac.findFundamentalMatrix(features[epoch]['mkpts0'], features[epoch]['mkpts1'], px_th=3, conf=0.9,
                                                          max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson',
                                                          symmetric_error_check=True, enable_degeneracy_check=True)
            F_matrix.append(F)
            print('Matches at epoch {}: pydegensac found {} inliers ({:.2f}%)'.format(epoch, inlMask.sum(),
                            inlMask.sum()*100 / len(features[epoch]['mkpts0'])))

        # Write matched points to disk   
        stem0, stem1 = Path(images[0][epoch]).stem, Path(images[1][epoch]).stem
        np.savetxt(os.path.join(epochdir, stem0+'_matchedPts.txt'), 
                   features[epoch]['mkpts0'] , fmt='%i', delimiter=',', newline='\n',
                   header='x,y') 
        np.savetxt(os.path.join(epochdir, stem1+'_matchedPts.txt'), 
                   features[epoch]['mkpts1'] , fmt='%i', delimiter=' ', newline='\n',                   
                   header='x,y') 
        with open(os.path.join(epochdir, stem0+'_'+stem1+'_features.pickle'), 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Matching completed')

elif not features: 
    epoch = 0
    matches_path = 'res/epoch_0/IMG_0520_IMG_2131_features.pickle'
    with open(matches_path, 'rb') as f:
        features = pickle.load(f)
        print("Loaded previous matches")
else:
    print("Features already present, nothing was changed.")

#%% SfM

#--- Realtive Pose with Essential Matrix ---#
pts0, pts1 = features[0]['mkpts0'], features[0]['mkpts1']
rel_pose = estimate_pose(pts0, pts1, cameras[0]['K'],  cameras[1]['K'], thresh=1, conf=0.99999)
R = rel_pose[0]
t = rel_pose[1]
valid = rel_pose[2]
print('Computing relative pose. Valid points: {}/{}'.format(valid.sum(),len(valid)))

# Build cameras structures
cameras[0]['R'], cameras[0]['t'] = np.eye(3), np.zeros((3,1))
cameras[1]['R'], cameras[1]['t'] = R, t.reshape(3,1)
for jj in range(0,numCams):
    cameras[jj]['P'] = P_from_KRT(cameras[jj]['K'], cameras[jj]['R'], cameras[jj]['t'])
    cameras[jj]['X0'] = X0_from_P(cameras[jj]['P'])

# Scale model by using camera baseline
X01_meta = np.array([416651.52489669225,5091109.91215075,1858.908434299682])   # IMG_2092
X02_meta = np.array([416622.27552777925,5091364.507128085,1902.4053286545502]) # IMG_0481
camWorldBaseline = np.linalg.norm(X01_meta - X02_meta)                         # [m] From Metashape model at epoch t0
camRelOriBaseline = np.linalg.norm(cameras[0]['X0'] - cameras[1]['X0'])
scaleFct = camWorldBaseline / camRelOriBaseline
cameras[1]['X0'] =  cameras[1]['X0'] * scaleFct
cameras[1]['t'] = -np.matmul(cameras[1]['R'], cameras[1]['X0'])
cameras[1]['P'] = P_from_KRT(cameras[1]['K'], cameras[1]['R'], cameras[1]['t'])

#--- Triangulate Points ---#
pts0_und = cv2.undistortPoints(features[0]['mkpts0'], cameras[0]['K'], cameras[0]['dist'], None, cameras[0]['K'])
pts1_und = cv2.undistortPoints(features[0]['mkpts1'], cameras[1]['K'], cameras[1]['dist'], None, cameras[1]['K'])
M, status = iterative_LS_triangulation(pts0_und, cameras[0]['P'],  pts1_und, cameras[1]['P'])
points3d.insert(epoch, M)
print(f'Triangulated success: {status.sum()/status.size}')

# Interpolate colors from image 
jj = 1
image = cv2.cvtColor(cv2.imread(images[jj][0], flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
points3d_cols =  interpolate_point_colors(points3d[0], image, cameras[jj]['P'], cameras[jj]['K'], cameras[jj]['dist'])

# Visualize and export sparse point cloud
do_viz = True

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points3d[epoch])
pcd.colors = o3d.utility.Vector3dVector(points3d_cols)
o3d.io.write_point_cloud("res/epoch_0/sparsepts_t"+str(epoch)+".ply", pcd)
if do_viz:
    o3d.visualization.draw_geometries([pcd])


#%% BUILD DSM AND ORTHOPHOTOS

## Build approximate DSM
dsm = build_dsm(points3d[epoch], dsm_step=0.1, save_path="sfm/dsm_approx.tif")

# Generate Ortophotos
from src.geometry import (P_from_KRT, project_points)

def generate_ortophoto(image, dsm, P, res=1):
    xx = dsm.x
    yy = dsm.y
    zz = dsm.z
    
    dsm_shape = dsm.x.shape
    ncell = dsm_shape[0]*dsm_shape[1]
    xyz = np.zeros((ncell,3))
    xyz[:,0] = xx.flatten()
    xyz[:,1] = yy.flatten()
    xyz[:,2] = zz.flatten()
    valid_cell = np.invert(np.isnan(xyz[:,2]))
    
    cols = np.full((ncell,3),0)
    cols[valid_cell,:] = interpolate_point_colors(xyz[valid_cell,:], image, camera['P'], camera['K'], camera['dist'])
    ortophoto = np.zeros((dsm_shape[0],dsm_shape[1],3))
    ortophoto[:,:,0] = cols[:,0].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto[:,:,1] = cols[:,1].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto[:,:,2] = cols[:,2].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto = np.uint8(ortophoto*255)
    
    # for a, b, c in zip(xx.flatten(), yy.flatten(), zz.flatten()):
    #     xyz.append([a, b, c]) 
    # ortophoto = cols[:,0].reshape(dsm_shape[0], dsm_shape[1],3)
    # ortophoto = np.uint8(ortophoto*255)

    return xyz, cols, ortophoto

    # ortophoto = None
    # return ortophoto


#%% DENSE MATCHING

# Init
sgm_path = Path('sgm')
downsample = 0.25
fast_viz = True

stem0 = Path(images[0][0]).stem
stem1 = Path(images[1][0]).stem

pts0, pts1 = features[0]['mkpts0'], features[0]['mkpts1']
F, inlMask = pydegensac.findFundamentalMatrix(pts0, pts1, px_th=1, conf=0.99999,
                                              max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson',
                                              symmetric_error_check=True, enable_degeneracy_check=True)

img0 = cv2.imread(images[0][0], flags=cv2.IMREAD_COLOR)
img1 = cv2.imread(images[1][0], flags=cv2.IMREAD_COLOR)
h, w, _ = img0.shape

#--- Rectify calibrated ---#
R1,R2,P1,P2,Q = cv2.stereoRectify(cameras[0]['K'], cameras[0]['dist'], cameras[1]['K'], cameras[1]['dist'],\
             (h,w), cameras[1]['R'], cameras[1]['t'])
    
    
#--- Recify Uncalibrated ---#

# undistort images
name0 = str(sgm_path / 'und' / (stem0 + "_undistorted.jpg"))
name1 = str(sgm_path / 'und' / (stem1 + "_undistorted.jpg"))
img0, K0_scaled = undistort_image(img0, cameras[0]['K'],  cameras[0]['dist'], downsample, name0)
img1, K1_scaled = undistort_image(img1, cameras[1]['K'],  cameras[1]['dist'], downsample, name1)

# Rectify uncalibrated
pts0, pts1 = features[0]['mkpts1']*downsample, features[0]['mkpts0']*downsample
F, inlMask = pydegensac.findFundamentalMatrix(pts0, pts1, px_th=1, conf=0.99999,
                                              max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson',
                                              symmetric_error_check=True, enable_degeneracy_check=True)
print('Pydegensac: {} inliers ({:.2f}%)'.format(inlMask.sum(), inlMask.sum()*100 / len(pts0)))
success, H1, H0 = cv2.stereoRectifyUncalibrated(pts0, pts1 , F, (w,h))
img0_rectified = cv2.warpPerspective(img0, H0, (w,h))
img1_rectified = cv2.warpPerspective(img1, H1, (w,h))

# write images to disk
path0 = str(sgm_path / 'rectified' / (stem0 + "_rectified.jpg"))
path1 = str(sgm_path / 'rectified' / (stem1 + "_rectified.jpg"))
cv2.imwrite(path0, img0_rectified)
cv2.imwrite(path1, img1_rectified)

if fast_viz: 
    print()
else:
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img0_rectified, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2RGB))
    plt.show()
    
    
#--- Run PSMNet to compute disparity ---#



#--- Find epilines corresponding to points in right image (second image) and drawing its lines on left image ---#
img0 = cv2.imread(images[0][0], flags=cv2.IMREAD_COLOR)
img1 = cv2.imread(images[1][0], flags=cv2.IMREAD_COLOR)

lines0 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 2, F)
lines0 = lines0.reshape(-1,3)
img0_epiplines, _ = draw_epip_lines(img0,img1,lines0,pts0,pts1)

# Find epilines corresponding to points in left image (first image) and drawing its lines on right image
lines1 = cv2.computeCorrespondEpilines(pts0.reshape(-1,1,2), 1, F)
lines1 = lines1.reshape(-1,3)
img1_epiplines,_ = draw_epip_lines(img1,img0,lines1,pts1,pts0, fast_viz=True)

if fast_viz:
    cv2.imwrite(str(sgm_path / (stem0 + "_epiplines.jpg")), img0_epiplines)
    cv2.imwrite(str(sgm_path / (stem1 + "_epiplines.jpg")), img1_epiplines)
else: 
    plt.subplot(121),plt.imshow(img0_epiplines)
    plt.subplot(122),plt.imshow(img1_epiplines)
    plt.show()
    
#--- Draw keypoints and matches ---#
pts0_rect = cv2.perspectiveTransform(np.float32(pts0).reshape(-1,1,2), H0).reshape(-1,2)
pts1_rect = cv2.perspectiveTransform(np.float32(pts1).reshape(-1,1,2), H1).reshape(-1,2)

# img0_rect_kpts = img0.copy()
img0 = cv2.imread(images[0][0], flags=cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(images[1][0], flags=cv2.IMREAD_GRAYSCALE)
pts0, pts1 = features[0]['mkpts0'], features[0]['mkpts1']
img0_kpts = cv2.drawKeypoints(img0,cv2.KeyPoint.convert(pts0),img0,(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img1_kpts = cv2.drawKeypoints(img1,cv2.KeyPoint.convert(pts1),img1,(0,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('kpts0.jpg', img0_kpts)
cv2.imwrite('kpts1.jpg', img1_kpts)

make_matching_plot(img0, img1, pts0, pts1, path='matches.jpg')        
