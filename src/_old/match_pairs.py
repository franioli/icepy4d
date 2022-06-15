from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import  pydegensac
from copy import deepcopy
import cv2 

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, frame2tensor,
                          vizTileRes)
from  models.tiles import (subdivideImage, 
                           appendPred, applyMatchesOffset)
torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='t0/impairs.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='t0/img/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='t0/res/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[-1],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true', default = True,
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=4096,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.0001,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=3,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=100,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.3,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true', default=True,
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',  default=True,
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true', default=False,
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',default=False,
        help='Force pytorch to run in CPU mode.')

    parser.add_argument(
        '--equalize_hist', action='store_true', default=False,
        help='Perform histogram equalization before feature extraction.')
    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    if opt.eval:
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(opt.input_pairs))

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    
    timer = AverageTimer(newline=True)
    for i, pair in enumerate(pairs):
        # i = 0
        # pair = pairs[i]
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
        viz_eval_path = output_dir / \
            '{}_{}_evaluation.{}'.format(stem0, stem1, opt.viz_extension)

        # Handle --cache logic.
        do_match = True
        do_eval = opt.eval
        do_viz = opt.viz
        do_viz_eval = opt.eval and opt.viz
        if opt.cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError('Cannot load matches .npz file: %s' %
                                  matches_path)

                kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                matches, conf = results['matches'], results['match_confidence']
                do_match = False
            #Removed opt.eval
            if opt.viz and viz_path.exists():
                do_viz = False
            if opt.viz and opt.eval and viz_eval_path.exists():
                do_viz_eval = False
            timer.update('load_cache')

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        image0, inp0, scales0 = read_image(
            input_dir / name0, device, opt.resize, rot0, opt.resize_float, opt.equalize_hist)
        image1, inp1, scales1 = read_image(
            input_dir / name1, device, opt.resize, rot1, opt.resize_float, opt.equalize_hist)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        timer.update('load_image')
        
        #%%
        useTile = True
        writeTile2Disk = True
        do_viz_tile = True
        rowDivisor = 2
        colDivisor = 3
        
        if useTile:
            timerTile = AverageTimer(newline=True)

            # Subdivide image in tiles and run a loop 
            print('Subdivinding image in tiles...')
            tiles0, limits0 = subdivideImage(image0, rowDivisor=rowDivisor, colDivisor=colDivisor, overlap = 200, 
                                              viz=False, out_dir=opt.input_dir+'tiles0', writeTile2Disk=False)
            tiles1, limits1 = subdivideImage(image1, rowDivisor=rowDivisor, colDivisor=colDivisor, overlap = 200, 
                                              viz=False, out_dir=opt.input_dir+'tiles1', writeTile2Disk=False)                                              
            timer.update('create_tiles')
            
            for t0, tile0 in enumerate(tiles0):
                # for t1, tile1 in enumerate(tiles1): 
                # Perform the matching.
                t1 = t0;
                inp0 = frame2tensor(tiles0[t0], device)
                inp1 = frame2tensor(tiles1[t1], device)   
                predTile = matching({'image0': inp0, 'image1': inp1})
                predTile = {k: v[0].cpu().numpy() for k, v in predTile.items()}   
                timerTile.update('matcher')
                
                kpts0, kpts1 = predTile['keypoints0'], predTile['keypoints1']
                descriptors0, descriptors1 = predTile['descriptors0'], predTile['descriptors1']    
                scores0, scores1 = predTile['scores0'], predTile['scores1']        
                matches0, matches1 = predTile['matches0'], predTile['matches1'],
                conf = predTile['matching_scores0']
                valid = matches0 > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches0[valid]]                    
                descriptors0  = descriptors0[:, valid]
                descriptors1  = descriptors1[:, matches0[valid]]
                scores0 = scores0[valid]
                scores1 = scores1[matches0[valid]]
                conf = conf[valid]
                
                if t0 < 1 and t1 < 1:
                    mkpts0_full = mkpts0.copy()
                    mkpts1_full = mkpts1.copy()
                    descriptors0_full = descriptors0.copy()
                    descriptors1_full = descriptors1.copy()
                    scores0_full = scores0.copy()
                    scores1_full = scores1.copy()
                    conf_full = conf.copy()
                else:              
                    mkpts0_full = np.append(mkpts0_full, mkpts0 + np.array(limits0[t0][0:2]).astype('float32'), axis=0)
                    mkpts1_full = np.append(mkpts1_full, mkpts1 + np.array(limits1[t1][0:2]).astype('float32'), axis=0)
                    scores0_full = np.append(scores0_full, scores0, axis=0)
                    scores1_full = np.append(scores1_full, scores1, axis=0)
                    conf_full =  np.append(conf_full, conf, axis=0)
                    descriptors0_full = np.append(descriptors0_full, descriptors0, axis=1)
                    descriptors1_full = np.append(descriptors1_full, descriptors1, axis=1) 
                
                if do_viz_tile:
                    vizTile_path = output_dir / '{}_{}_matches_tile{}_{}.{}'.format(stem0, stem1, t0, t1, opt.viz_extension)
                    tile_print_opt = {'imstem0': stem0+'_'+str(t0), 'imstem1': stem1+'_'+str(t1), 'show_keypoints': True, 
                           'fast_viz': opt.fast_viz, 'opencv_display': opt.opencv_display}
                    vizTileRes(vizTile_path, predTile, tiles0[t0], tiles1[t1], matching, timerTile, tile_print_opt)     
                    
                timerTile.print('Finished Tile Pairs {:2} - {:2} of {:2}'.format(t0, t1, len(tiles0)*len(tiles1)))

             

        # Restore original image coordinates (not cropped) and run PyDegensac
        maskBB = np.array([[600,1900,4700,1700], [800,1800,4700,1700]]).astype('float32')  
    
        mkpts0_full = mkpts0_full + np.array(maskBB[0][0:2]).astype('float32')
        mkpts1_full = mkpts1_full + np.array(maskBB[1][0:2]).astype('float32')
        
        F, inlMask = pydegensac.findFundamentalMatrix(mkpts0_full, mkpts1_full, px_th=2, conf=0.9999, 
                                                      max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson', 
                                                      symmetric_error_check=True, enable_degeneracy_check=True)
        print ('pydegensac found {} inliers ({:.2f}%)'.format(int(deepcopy(inlMask).astype(np.float32).sum()),
                        int(deepcopy(inlMask).astype(np.float32).sum())*100 / len(mkpts0_full)) )
        mconf = conf_full[inlMask]
        mkpts0 = mkpts0_full[inlMask]
        mkpts1 = mkpts1_full[inlMask]
        scores0 = scores0_full[inlMask]
        scores1 = scores1_full[inlMask]
        descriptors0 = descriptors0_full[:,inlMask]
        descriptors1 = descriptors1_full[:,inlMask]
        timer.update('PyDegensac')

        # Write the matches to disk.
        out_matches = {'mkpts0': mkpts0 , 'mkpts1': mkpts1,
               'matches': matches0, 'match_confidence': conf}        
        np.savez(str(matches_path), **out_matches)
    
        
        # Write tensors to disk
        prev0 = {}
        prev0['keypoints0'] = [torch.from_numpy(mkpts0).to(device)]
        prev0['scores0'] = (torch.from_numpy(scores0).to(device),)
        prev0['descriptors0'] = [torch.from_numpy(descriptors0).to(device)]
        torch.save(prev0, str(matches_path.parent / '{}_tensor_0.pt'.format(matches_path.stem)))
        
        prev1 = {}
        prev1['keypoints0'] = [torch.from_numpy(mkpts1).to(device)]
        prev1['scores0'] = (torch.from_numpy(scores1).to(device),)
        prev1['descriptors0'] = [torch.from_numpy(descriptors1).to(device)]
        torch.save(prev1, str(matches_path.parent / '{}_tensor_1.pt'.format(matches_path.stem)))
    
    
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
                image0, image1, mkpts0-maskBB[0][0:2], mkpts1-maskBB[1][0:2], 
                mkpts0-maskBB[0][0:2], mkpts1-maskBB[1][0:2], color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)
    
            timer.update('viz_match')
        
    timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

#%% Recover full image coordinates (before crop)

# maskBB = np.array([[600,1900,4700,1700], [800,1800,4700,1700]])


# mkpts0_full = mkpts0_full + np.array(maskBB[0][0:2]).astype('float32')
# mkpts1_full = mkpts1_full + np.array(maskBB[1][0:2]).astype('float32')

# F, inlMask = pydegensac.findFundamentalMatrix(mkpts0_full, mkpts1_full, px_th=2, conf=0.9999, 
#                                               max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson', 
#                                               symmetric_error_check=True, enable_degeneracy_check=True)
# print ('pydegensac found {} inliers ({:.2f}%)'.format(int(deepcopy(inlMask).astype(np.float32).sum()),
#                 int(deepcopy(inlMask).astype(np.float32).sum())*100 / len(mkpts0_full)) )
# mconf = conf_full[inlMask]
# mkpts0 = mkpts0_full[inlMask]
# mkpts1 = mkpts1_full[inlMask]
# scores0 = scores0_full[inlMask]
# scores1 = scores1_full[inlMask]
# descriptors0 = descriptors0_full[:,inlMask]
# descriptors1 = descriptors1_full[:,inlMask]
# timer.update('PyDegensac')


#%% Reconstruct Geometry
