from pathlib import Path
# import argparse
import numpy as np
import matplotlib.cm as cm
import torch
import  pydegensac
from copy import deepcopy
import cv2 

from utils.sg.matching import Matching
from utils.sg.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, frame2tensor,
                          vizTileRes)
from  utils.utils import generateTiles
torch.set_grad_enabled(False)

def match_pair(pair, maskBB, opt):
    
    assert not (opt['opencv_display'] and not opt['viz']), 'Must use --viz with --opencv_display'
    assert not (opt['opencv_display'] and not opt['fast_viz']), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt['fast_viz'] and not opt['viz']), 'Must use --viz with --fast_viz'
    assert not (opt['fast_viz'] and opt['viz_extension'] == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    if len(opt['resize']) == 2 and opt['resize'][1] == -1:
        opt['resize'] = opt['resize'][0:1]
    if len(opt['resize']) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt['resize'][0], opt['resize'][1]))
    elif len(opt['resize']) == 1 and opt['resize'][0] > 0:
        print('Will resize max dimension to {}'.format(opt['resize'][0]))
    elif len(opt['resize']) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt['force_cpu'] else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt['nms_radius'],
            'keypoint_threshold': opt['keypoint_threshold'],
            'max_keypoints': opt['max_keypoints']
        },
        'superglue': {
            'weights': opt['superglue'],
            'sinkhorn_iterations': opt['sinkhorn_iterations'],
            'match_threshold': opt['match_threshold'],
        }
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    output_dir = Path(opt['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt['viz']:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))
    
    timer = AverageTimer(newline=True)
    name0, name1 = pair[:2]
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
    viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt['viz_extension'])
 

    # Load the image pair.
    # If a rotation integer is provided (e.g. from EXIF data), use it: ! Not working
    # if len(pair) >= 5:
    #     rot0, rot1 = int(pair[2]), int(pair[3])
    # else:
    rot0, rot1 = 0, 0
    image0, inp0, scales0 = read_image(
        name0, device, opt['resize'], rot0, opt['resize_float'], maskBB[0], opt['equalize_hist'])
    image1, inp1, scales1 = read_image(
        name1, device, opt['resize'], rot1, opt['resize_float'], maskBB[1], opt['equalize_hist'])
    if image0 is None or image1 is None:
        print('Problem reading image pair: {} {}'.format(
            name0, name1))
        exit(1)
    timer.update('load_image')
    # import matplotlib
    # matplotlib.use('Qt5Agg')
    # import matplotlib.pyplot as plt
    # plt.imshow(cv2.cvtColor(image1/255., cv2.COLOR_BGR2RGB))
    # plt.show()  
    
    do_viz = opt['viz']
    useTile = opt['useTile']
    writeTile2Disk = opt['writeTile2Disk']
    do_viz_tile = opt['do_viz_tile']
    rowDivisor = opt['rowDivisor']
    colDivisor = opt['colDivisor']
    overlap = opt['overlap']
    
    if useTile:
        timerTile = AverageTimer(newline=True)

        # Subdivide image in tiles and run a loop            
        tiles0, limits0 = generateTiles(image0, rowDivisor=rowDivisor, colDivisor=colDivisor, overlap = overlap, 
                                          viz=do_viz_tile, out_dir=output_dir/'tiles0', writeTile2Disk=writeTile2Disk)
        tiles1, limits1 = generateTiles(image1, rowDivisor=rowDivisor, colDivisor=colDivisor, overlap = overlap, 
                                          viz=do_viz_tile, out_dir=output_dir/'tiles1', writeTile2Disk=writeTile2Disk)         
        print(f'Images subdivided in {rowDivisor}x{colDivisor} tiles')                                     
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
                vizTile_path = output_dir / '{}_{}_matches_tile{}_{}.{}'.format(stem0, stem1, t0, t1, opt['viz_extension'])
                tile_print_opt = {'imstem0': stem0+'_'+str(t0), 'imstem1': stem1+'_'+str(t1), 'show_keypoints': True, 
                       'fast_viz': opt['fast_viz'], 'opencv_display': opt['opencv_display']}
                vizTileRes(vizTile_path, predTile, tiles0[t0], tiles1[t1], matching, timerTile, tile_print_opt)     
                
            timerTile.print('Finished Tile Pairs {:2} - {:2} of {:2}'.format(t0, t1, len(tiles0)))

         

    # Restore original image coordinates (not cropped) and run PyDegensac    
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
    out_matches = {'mkpts0': mkpts0 , 'mkpts1': mkpts1, 'match_confidence': mconf}        
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
            image0, image1, mkpts0-maskBB[0][0:2].astype('float32'), mkpts1-maskBB[1][0:2].astype('float32'), 
            mkpts0-maskBB[0][0:2].astype('float32'), mkpts1-maskBB[1][0:2].astype('float32'), color,
            text, viz_path, opt['show_keypoints'],
            opt['fast_viz'], opt['opencv_display'], 'Matches', small_text)

        timer.update('viz_match')
    
    timer.print('Finished pair')

    # Free cuda memory and return variables
    torch.cuda.empty_cache()
    return out_matches, [descriptors0, descriptors1], [scores0, scores1], [prev0, prev1]