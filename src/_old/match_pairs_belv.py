#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import os

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
torch.set_grad_enabled(False)

#%% Parameters

input_pairs = 'impairs.txt'                                   
input_dir = 'img'
output_dir = 'res'

max_length = -1
resize = [1600]  #1600
resize_float = True
equalize_hist = False

# Superpoing
max_keypoints = 4096
keypoint_threshold = 0.05
nms_radius = 3

# SuperGlue
superglue = 'outdoor'
match_threshold = 0.3
sinkhorn_iterations = 100

# Visualization
viz = True
fast_viz = True
show_keypoints = False
viz_extension = 'png'
opencv_display = False

# Others
eval = False
cache = False
shuffle = False
force_cpu = False

#%% Read impairs.txt file

if len(resize) == 2 and resize[1] == -1:
    resize = resize[0:1]
if len(resize) == 2:
    print('Will resize to {}x{} (WxH)'.format(
        resize[0], resize[1]))
elif len(resize) == 1 and resize[0] > 0:
    print('Will resize max dimension to {}'.format(resize[0]))
elif len(resize) == 1:
    print('Will not resize images')
else:
    raise ValueError('Cannot specify more than two integers for --resize')

with open(input_pairs, 'r') as f:
    pairs = [l.split() for l in f.readlines()]

if max_length > -1:
    pairs = pairs[0:np.min([len(pairs), max_length])]

if shuffle:
    random.Random(0).shuffle(pairs)
    
# Create the output directories if they do not exist already.
input_dir = Path(input_dir)
print('Looking for data in directory \"{}\"'.format(input_dir))
output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
print('Will write matches to directory \"{}\"'.format(output_dir))
if eval:
    print('Will write evaluation results',
          'to directory \"{}\"'.format(output_dir))
if viz:
    print('Will write visualization images to',
          'directory \"{}\"'.format(output_dir))
    
timer = AverageTimer(newline=True)



#%%================================


#%% Load the SuperPoint and SuperGlue models.
device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': nms_radius,
        'keypoint_threshold': keypoint_threshold,
        'max_keypoints': max_keypoints
    },
    'superglue': {
        'weights': superglue,
        'sinkhorn_iterations': sinkhorn_iterations,
        'match_threshold': match_threshold,
    }
}
matching = Matching(config).eval().to(device)

#%% Read data

# using these values istead of loop
i = 0
pair = pairs[i]
print(f'Matching images {pair}')

name0, name1 = pair[:2]
stem0, stem1 = Path(name0).stem, Path(name1).stem
matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, viz_extension)
viz_eval_path = output_dir / \
    '{}_{}_evaluation.{}'.format(stem0, stem1, viz_extension)  

# Handle --cache logic.
do_match = True
do_eval = eval
do_viz = viz
do_viz_eval = eval and viz
    
if not (do_match or do_eval or do_viz or do_viz_eval):
    timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs))) 
    
# If a rotation integer is provided (e.g. from EXIF data), use it:
if len(pair) >= 5:
    rot0, rot1 = int(pair[2]), int(pair[3])
else:
    rot0, rot1 = 0, 0    

# Load the image pair.
image0, inp0, scales0 = read_image(
    input_dir / name0, device, resize, rot0, resize_float, equalize_hist)
image1, inp1, scales1 = read_image(
    input_dir / name1, device, resize, rot1, resize_float, equalize_hist)
if image0 is None or image1 is None:
    print('Problem reading image pair: {} {}'.format(
        input_dir/name0, input_dir/name1))
    exit(1)
timer.update('load_image')

#%%  Perform the matching.

pred = matching({'image0': inp0, 'image1': inp1})
torch.save(pred, str(matches_path.parent / '{}_tensor.pt'.format(matches_path.stem)))

#%% Store results
# pred_t0 = pred
pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
matches, conf = pred['matches0'], pred['matching_scores0']
timer.update('matcher')

# Keep the matching keypoints.
valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]

# Write the matches to disk.
out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
               'matches': matches, 'match_confidence': conf, 
               'mkpts0': mkpts0 , 'mkpts1': mkpts1}
np.savez(str(matches_path), **out_matches)
np.savez(str(matches_path.with_suffix(''))+'_full.npz', **pred)

# Visualize and/or save results
if do_viz:
    # Visualize the matches.
    color = cm.jet(mconf)
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]
    if rot0 != 0 or rot1 != 0:
        text.append('Rotation: {}:{}'.format(rot0, rot1))

    # Display extra parameter info.
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {}:{}'.format(stem0, stem1),
    ]

    make_matching_plot(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
        text, viz_path, show_keypoints,
        fast_viz, opencv_display, 'Matches', small_text)

    timer.update('viz_match')
    
timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

#%%================================

#%% Track matched features on previous epoch in the next epochs

# Overwrite parameters
equalize_hist = False
max_keypoints = 4096
keypoint_threshold = 0.005
nms_radius = 3

match_threshold = 0.3
sinkhorn_iterations = 100

# Load the SuperPoint and SuperGlue models.
device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': nms_radius,
        'keypoint_threshold': keypoint_threshold,
        'max_keypoints': max_keypoints
    },
    'superglue': {
        'weights': superglue,
        'sinkhorn_iterations': sinkhorn_iterations,
        'match_threshold': match_threshold,
    }
}
matching = Matching(config).eval().to(device)
    
#%% Track features on p1

i = 0
pair = pairs[i]
prev_epoch_dir = './t0/res'

print(f'Tracking features at next epoch. Processing image pair: {pair}...')


# Load features matched at previous epoch to be used ad keypoints
def selectTensorKpts(a, valid_rows, device):
    n = a.cpu().detach().numpy()
    n = n[valid]
    n = torch.from_numpy(n).to(device)
    return n

def selectTensorDescr(a, valid_rows, device):
    n = a.cpu().detach().numpy()
    n = n[:,valid]
    n = torch.from_numpy(n).to(device)
    return n

       
name0, name1 = pair[:2]
stem0, stem1 = Path(name0).stem, Path(name1).stem
matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, viz_extension)
viz_eval_path = output_dir / \
    '{}_{}_evaluation.{}'.format(stem0, stem1, viz_extension)     

# If a rotation integer is provided (e.g. from EXIF data), use it:
if len(pair) >= 5:
    rot0, rot1 = int(pair[2]), int(pair[3])
else:
    rot0, rot1 = 0, 0

# Load the image pair.
image0, inp0, scales0 = read_image(
    input_dir / name0, device, resize, rot0, resize_float, equalize_hist)
image1, inp1, scales1 = read_image(
    input_dir / name1, device, resize, rot1, resize_float, equalize_hist)
if image0 is None or image1 is None:
    print('Problem reading image pair: {} {}'.format(
        input_dir/name0, input_dir/name1))
    exit(1)
timer.update('load_image')


# Load matching from previous epoch 
for file in os.listdir(prev_epoch_dir):
    if file.endswith(".pt"):
        last_data_path = os.path.join(prev_epoch_dir, file)
keys = ['keypoints0', 'scores0', 'descriptors0']
last_data = torch.load(last_data_path)
# last_data 
matches = {k: v[0].cpu().numpy() for k, v in last_data.items()}['matches0']
valid = matches > -1
last_data = {k: last_data[k]  for k in keys}
last_data['keypoints0'] = [selectTensorKpts(last_data['keypoints0'][0], valid, device)]
last_data['scores0'] = (selectTensorKpts(last_data['scores0'][0], valid, device),)
last_data['descriptors0'] = [selectTensorDescr(last_data['descriptors0'][0], valid, device)]
last_data['image0'] = inp0

# Perform matching
pred = matching({**last_data, 'image1': inp1})

# Save results
pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
kpts0 = last_data['keypoints0'][0].cpu().numpy()
kpts1 =  pred['keypoints1']
matches, conf = pred['matches0'], pred['matching_scores0']
timer.update('matcher')
    
# Write the matches to disk.
out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
               'matches': matches, 'match_confidence': conf}
np.savez(str(matches_path), **out_matches)
np.savez(str(matches_path.with_suffix(''))+'_full.npz', **pred)

# Keep the matching keypoints.
valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]

# Visualize the matches.
color = cm.jet(mconf)
text = [
    'SuperGlue',
    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
    'Matches: {}'.format(len(mkpts0)),
]
if rot0 != 0 or rot1 != 0:
    text.append('Rotation: {}:{}'.format(rot0, rot1))

# Display extra parameter info.
k_thresh = matching.superpoint.config['keypoint_threshold']
m_thresh = matching.superglue.config['match_threshold']
small_text = [
    'Keypoint Threshold: {:.4f}'.format(k_thresh),
    'Match Threshold: {:.2f}'.format(m_thresh),
    'Image Pair: {}:{}'.format(stem0, stem1),
]

make_matching_plot(
    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
    text, viz_path, show_keypoints,
    fast_viz, opencv_display, 'Matches', small_text)

timer.update('viz_match')

timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

#%% Track features on p2



#%% Track matched features on previous epoch in the next epochs

# Overwrite parameters
equalize_hist = False
max_keypoints = 4096
keypoint_threshold = 0.05
nms_radius = 3

match_threshold = 0.3
sinkhorn_iterations = 100

# Load the SuperPoint and SuperGlue models.
device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': nms_radius,
        'keypoint_threshold': keypoint_threshold,
        'max_keypoints': max_keypoints
    },
    'superglue': {
        'weights': superglue,
        'sinkhorn_iterations': sinkhorn_iterations,
        'match_threshold': match_threshold,
    }
}
matching = Matching(config).eval().to(device)
    

# Load features matched at previous epoch to be used ad keypoints
def selectTensorKpts(a, valid_rows, device):
    n = a.cpu().detach().numpy()
    n = n[valid]
    n = torch.from_numpy(n).to(device)
    return n

def selectTensorDescr(a, valid_rows, device):
    n = a.cpu().detach().numpy()
    n = n[:,valid]
    n = torch.from_numpy(n).to(device)
    return n


i = 0
pair = pairs[i]
print(f'Tracking features at next epoch. Processing image pair: {pair}...')
prev_epoch_dir = 'D:/francesco/belvedereStereo/SGmatching/t0/res'


name0, name1 = pair[:2]
stem0, stem1 = Path(name0).stem, Path(name1).stem
matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, viz_extension)
viz_eval_path = output_dir / \
    '{}_{}_evaluation.{}'.format(stem0, stem1, viz_extension)     

# If a rotation integer is provided (e.g. from EXIF data), use it:
if len(pair) >= 5:
    rot0, rot1 = int(pair[2]), int(pair[3])
else:
    rot0, rot1 = 0, 0

# Load the image pair.
image0, inp0, scales0 = read_image(
    input_dir / name0, device, resize, rot0, resize_float, equalize_hist)
image1, inp1, scales1 = read_image(
    input_dir / name1, device, resize, rot1, resize_float, equalize_hist)
if image0 is None or image1 is None:
    print('Problem reading image pair: {} {}'.format(
        input_dir/name0, input_dir/name1))
    exit(1)
timer.update('load_image')

# Load matching from previous epoch 
for file in os.listdir(prev_epoch_dir):
    if file.endswith(".pt"):
        last_data_path = os.path.join(prev_epoch_dir, file)
keys = ['keypoints0', 'scores0', 'descriptors0']
prev = torch.load(last_data_path)
matches = {k: v[0].cpu().numpy() for k, v in prev.items()}['matches1']
valid = matches > -1
last_data = {}
last_data['keypoints0'] = [selectTensorKpts(prev['keypoints1'][0],  valid, device)]
last_data['scores0'] = (selectTensorKpts(prev['scores1'][0], valid, device),)
last_data['descriptors0'] = [selectTensorDescr(prev['descriptors1'][0], valid, device)]
last_data['image0'] = inp0

# Perform matching
pred = matching({**last_data, 'image1': inp1})

# Save results
pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
kpts0 = last_data['keypoints0'][0].cpu().numpy()
kpts1 =  pred['keypoints1']
matches, conf = pred['matches0'], pred['matching_scores0']
timer.update('matcher')
    
# Write the matches to disk.
out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
               'matches': matches, 'match_confidence': conf}
np.savez(str(matches_path), **out_matches)
np.savez(str(matches_path.with_suffix(''))+'_full.npz', **pred)

# Keep the matching keypoints.
valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]

# Visualize the matches.
color = cm.jet(mconf)
text = [
    'SuperGlue',
    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
    'Matches: {}'.format(len(mkpts0)),
]
if rot0 != 0 or rot1 != 0:
    text.append('Rotation: {}:{}'.format(rot0, rot1))

# Display extra parameter info.
k_thresh = matching.superpoint.config['keypoint_threshold']
m_thresh = matching.superglue.config['match_threshold']
small_text = [
    'Keypoint Threshold: {:.4f}'.format(k_thresh),
    'Match Threshold: {:.2f}'.format(m_thresh),
    'Image Pair: {}:{}'.format(stem0, stem1),
]

make_matching_plot(
    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
    text, viz_path, show_keypoints,
    fast_viz, opencv_display, 'Matches', small_text)

timer.update('viz_match')

timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
