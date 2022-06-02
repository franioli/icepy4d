from pathlib import Path
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


def track_matches(pairs, maskBB, prevs, opt):
 
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

    # Inizialize lists for storing matching points
    kpts0_full = []
    kpts1_full = []
    wasMatched = []
    # mconf_full = []
    descriptors1_full = []
    scores1_full = []

    timer = AverageTimer(newline=True)
    for cam, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt['viz_extension'])

        # Load the image pair.
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
    
        # Load matches from previous epoch
        # prev = prevs[cam]
        # for file in os.listdir(opt.prev_epoch_dir):
        #     # if file.endswith(str(opt.cam_num)+".pt"):
        #     if file.endswith(str(cam)+".pt"):
        #         last_data_path = os.path.join(opt.prev_epoch_dir, file)
        # prev = torch.load(last_data_path)

        # Subdivide image in tiles
        do_viz = opt['viz']
        useTile = opt['useTile']
        writeTile2Disk = opt['writeTile2Disk']
        do_viz_tile = opt['do_viz_tile']
        rowDivisor = opt['rowDivisor']
        colDivisor = opt['colDivisor']

        timerTile = AverageTimer(newline=True)
        tiles0, limits0 = generateTiles(image0, rowDivisor=rowDivisor, colDivisor=colDivisor, overlap=0,
                                          viz=do_viz_tile, out_dir=output_dir/'tiles0', writeTile2Disk=writeTile2Disk)
        tiles1, limits1 = generateTiles(image1, rowDivisor=rowDivisor, colDivisor=colDivisor, overlap=0,
                                          viz=do_viz_tile, out_dir=output_dir/'tiles1', writeTile2Disk=writeTile2Disk)         
        print(f'Images subdivided in {rowDivisor}x{colDivisor} tiles')                                     
        timer.update('create_tiles')

#%% Run tracking
     
        # try:
        #     if torch.is_tensor(prevs[cam]['keypoints0'][0]):
        #         prev = {k: v[0].cpu().numpy() for k, v in prevs[cam].items()}   
        #     print('Prev data are tensors: converted to np')
        # except:        
        prev = prevs[cam]
        kpts0_full.append(np.full(np.shape(prev['keypoints0']), -1, dtype=(float)))
        kpts1_full.append(np.full(np.shape(prev['keypoints0']), -1, dtype=(float)))
        wasMatched.append(np.full( len(prev['keypoints0']), -1, dtype=(float) ))
        
        descriptors1_full.append(np.full( np.shape(prev['descriptors0']), -1, dtype=(float) ))
        scores1_full.append(np.full( len(prev['scores0']), -1, dtype=(float) ))
        # mconf_full.append(np.full((len(prev['keypoints0'])), 0, dtype=(float)))
       
        for t, tile0 in enumerate(tiles0):
            # Shift back to previuos keypoints
            def rectContains(rect, pt):
                logic = rect[0] < pt[0] < rect[2] and rect[1] < pt[1] < rect[3]
                return logic

            # Subract coordinates bounding box
            kpts0 = prev['keypoints0'] - np.array(maskBB[0][0:2]).astype('float32')
            
            # Keep only kpts in current tile
            ptsInTile = np.zeros(len(kpts0), dtype=(bool))
            for i, kk in enumerate(kpts0):
                ptsInTile[i] = rectContains(limits0[t], kk)
                ptsInTileIdx = np.where(ptsInTile == True)
            kpts0 = kpts0[ptsInTile] - np.array(limits0[t][0:2]).astype('float32')

            # Build Prev tensor
            prevTile = {'keypoints0': [], 'scores0': [], 'descriptors0': []}
            prevTile['keypoints0'] = kpts0
            prevTile['scores0'] = prev['scores0'][ptsInTile]
            prevTile['descriptors0'] = prev['descriptors0'][:, ptsInTile]
            # a = prevTile['keypoints0'].astype(int)
            # a_ = tiles0[t].copy()/255.
            # a_ = cv2.cvtColor(a_ , cv2.COLOR_GRAY2RGB);
            # for aa in a:
            #     a_ = cv2.drawMarker(a_, tuple(aa), (0, 255, 255))
            # cv2.imshow('test',a_)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # Perform the matching.
            inp0 = frame2tensor(tiles0[t], device)
            inp1 = frame2tensor(tiles1[t], device)
            prevTile = {k: [torch.from_numpy(v).to(device)]
                                             for k, v in prevTile.items()}
            prevTile['image0'] = inp0
            predTile = matching({**prevTile, 'image1': inp1})
            predTile = {k: v[0].cpu().numpy() for k, v in predTile.items()}
            timerTile.update('matcher')

            # Retrieve points
            kpts1 = predTile['keypoints1']
            descriptors1 = predTile['descriptors1']
            scores1 = predTile['scores1']
            matches0 = predTile['matches0']
            conf = predTile['matching_scores0']
            valid = matches0 > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches0[valid]]
            mconf = conf[valid]

            for i, pt in enumerate(kpts0):
                if predTile['matches0'][i] > -1:
                    wasMatched[cam][ptsInTileIdx[0][i]] = 1
                    kpts0_full[cam][ptsInTileIdx[0][i], :] = pt + np.array(limits0[t][0:2]).astype('float32')
                    kpts1_full[cam][ptsInTileIdx[0][i], :] = kpts1[predTile['matches0'][i]] + np.array(limits1[t][0:2]).astype('float32')      
                 
                    descriptors1_full[cam][:, ptsInTileIdx[0][i]] = descriptors1[:, predTile['matches0'][i]]            
                    scores1_full[cam][ptsInTileIdx[0][i]] = scores1[predTile['matches0'][i]]                
                    # mconf_full[cam][ptsInTileIdx[0][i], :] = predTile['matches0']
                    #     [i]] + np.array(limits1[t][0:2]).astype('float32') 
                    #TO DO: Keep track of the matching scores
                    
            if t < 1:         
                mconf_full = mconf.copy()
            else:
                mconf_full = np.append(mconf_full, mconf, axis=0)


            if do_viz_tile:
                # Visualize the matches.
                vizTile_path= output_dir / '{}_{}_matches_tile{}.{}'.format(stem0, stem1, t, opt['viz_extension'])
                color = cm.jet(mconf)
                text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0)),
                ]
                k_thresh = matching.superpoint.config['keypoint_threshold']
                m_thresh = matching.superglue.config['match_threshold']
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh),
                    'Image Pair: {}:{}'.format(stem0, stem1),
                ]
                make_matching_plot(
                    tiles0[t],  tiles1[t], kpts0, kpts1, mkpts0, mkpts1, color,
                    text, vizTile_path, True,
                    opt['fast_viz'], opt['opencv_display'], 'Matches', small_text)

            # if do_viz_tile:
            #     predTile['keypoints0']= kpts0
            #     vizTile_path= output_dir / '{}_{}_matches_tile{}.{}'.format(stem0, stem1, t, opt.viz_extension)
            #     tile_print_opt= {'imstem0': stem0+'_'+str(t), 'imstem1': stem1+'_'+str(t), 'show_keypoints': True,
            #            'fast_viz': opt.fast_viz, 'opencv_display': opt.opencv_display}
            #     vizTileRes(vizTile_path, predTile,
            #                tiles0[t], tiles1[t], matching, timerTile, tile_print_opt)

            timerTile.print(
                'Finished Tile Pairs {:2} of {:2}'.format(t, len(tiles0)))



        if do_viz:
            # Visualize the matches.
            val = wasMatched[cam] == 1
            color= cm.jet(mconf_full)  # mconf_full
            text= [
                'SuperGlue',
                'Keypoints: {}:{}'.format(
                    len(kpts0_full[cam]), len(kpts1_full[cam])),
                'Matches: {}'.format(len(kpts1_full[cam])),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh= matching.superpoint.config['keypoint_threshold']
            m_thresh= matching.superglue.config['match_threshold']
            small_text= [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, 
                kpts0_full[cam][val], kpts1_full[cam][val], 
                kpts0_full[cam][val], kpts1_full[cam][val], 
                color, text, viz_path, opt['show_keypoints'],
                opt['fast_viz'], opt['opencv_display'], 'Matches', small_text)

            timer.update('viz_match')

        
        timer.print('Finished pair {:5} of {:5}'.format(cam+1, len(pairs)))
        
        # torch.cuda.empty_cache()


    # %%

    # Retrieve points that were matched in both the images
    validTracked= [m == 2 for m in wasMatched[0] + wasMatched[1]]
    mkpts1_cam0 = kpts1_full[0][validTracked]
    mkpts1_cam1 = kpts1_full[1][validTracked]
    descr1_cam0  = descriptors1_full[0][:, validTracked]
    descr1_cam1  = descriptors1_full[1][:, validTracked]
    scores1_cam0 = scores1_full[0][validTracked]
    scores1_cam1 = scores1_full[1][validTracked]

    # Restore original image coordinates (not cropped) 
    mkpts1_cam0= mkpts1_cam0 + maskBB[0][0:2].astype('float32')
    mkpts1_cam1= mkpts1_cam1 + maskBB[1][0:2].astype('float32')

    # Run PyDegensac
    # F, inlMask= pydegensac.findFundamentalMatrix(mkpts1_cam0, mkpts1_cam1, px_th=3, conf=0.9,
    #                                               max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson',
    #                                               symmetric_error_check=True, enable_degeneracy_check=True)
    # print('pydegensac found {} inliers ({:.2f}%)'.format(int(deepcopy(inlMask).astype(np.float32).sum()),
    #                 int(deepcopy(inlMask).astype(np.float32).sum())*100 / len(mkpts1_cam0)))
    
    # # Reject false matching
    # mkpts1_cam0= mkpts1_cam0[inlMask]
    # mkpts1_cam1= mkpts1_cam1[inlMask]
    # descr1_cam0  = descr1_cam0[:, inlMask]
    # descr1_cam1  = descr1_cam1[:, inlMask]
    # scores1_cam0 = scores1_cam0[inlMask]
    # scores1_cam1 = scores1_cam1[inlMask]
    # timer.update('PyDegensac')

    # Viz point mached on both the images
    name0= pairs[0][1]
    name1= pairs[1][1]
    stem0, stem1= Path(name0).stem, Path(name1).stem
    viz_path= output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt['viz_extension'])
    matches_path= output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
    image0, _, _ = read_image(
         name0, device, opt['resize'], rot0, opt['resize_float'], maskBB[0], opt['equalize_hist'])
    image1, _, _ = read_image(
         name1, device, opt['resize'], rot0, opt['resize_float'], maskBB[0], opt['equalize_hist'])

    if do_viz:
        # Visualize the matches.
        color= cm.jet(mconf_full)
        text= [
            'SuperGlue',
            # 'Keypoints: {}:{}'.format(len(kpts0_full[cam]), len(kpts1_full[cam])),
            'Matches: {}'.format(len(mkpts1_cam1)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append('Rotation: {}:{}'.format(rot0, rot1))

        # Display extra parameter info.
        k_thresh= matching.superpoint.config['keypoint_threshold']
        m_thresh= matching.superglue.config['match_threshold']
        small_text= [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {}:{}'.format(stem0, stem1),
        ]

        make_matching_plot(
            image0, image1, mkpts1_cam0 - \
                maskBB[0][0:2], mkpts1_cam1-maskBB[1][0:2],
            mkpts1_cam0-maskBB[0][0:2], mkpts1_cam1-maskBB[1][0:2], color,
            text, viz_path, opt['show_keypoints'],
            opt['fast_viz'], opt['opencv_display'], 'Matches', small_text)

        timer.update('viz_match')

    # Write to Disk
    out_matches= {'mkpts0': mkpts1_cam0, 'mkpts1': mkpts1_cam1,
                'match_confidence': []}
    np.savez(str(matches_path), **out_matches)
    
    
    # Free cuda memory and return variables
    torch.cuda.empty_cache()
    
    tracked_cam0 = {'keypoints1': mkpts1_cam0, 'descriptors1': descr1_cam0, 'scores1': scores1_cam0}
    tracked_cam1 = {'keypoints1': mkpts1_cam1, 'descriptors1': descr1_cam1, 'scores1': scores1_cam1}

    return tracked_cam0, tracked_cam1