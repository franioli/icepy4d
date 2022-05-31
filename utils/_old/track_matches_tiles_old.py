from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import os
import cv2
import pydegensac
from copy import deepcopy

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, frame2tensor,
                          vizTileRes)
from models.tiles import (subdivideImage,
                           appendPred, applyMatchesOffset)
torch.set_grad_enabled(False)

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('qt5agg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='t1/from_t0/impairs.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='t1/from_t0/img/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='t1/from_t0/res/',
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
        '--resize_float', action='store_true', default=True,
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=8192,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.0001,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.5,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true', default=True,
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true', default=True,
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
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
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    parser.add_argument(
        '--prev_epoch_dir', type=str, default='t0/res/',
        help='Directory of the results of the previous epoch matching.')
    parser.add_argument(
        '--cam_num', type=int, default=0,
        help='Perform histogram equalization before feature extraction.')
    parser.add_argument(
        '--equalize_hist', action='store_true', default=False,
        help='Perform histogram equalization before feature extraction.')

    opt = parser.parse_args()
    print(opt)

    assert not (
        opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (
        opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension ==
                'pdf'), 'Cannot use pdf extension with --fast_viz'

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

    # Inizialize lists for storing matching points
    kpts0_full = []
    kpts1_full = []
    wasMatched = []
    # mconf_full = []

    timer = AverageTimer(newline=True)
    for cam, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        viz_path = output_dir / \
            '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)

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
            if opt.viz and viz_path.exists():
                do_viz = False
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

        # Load matches from previous epoch
        for file in os.listdir(opt.prev_epoch_dir):
            # if file.endswith(str(opt.cam_num)+".pt"):
            if file.endswith(str(cam)+".pt"):
                last_data_path = os.path.join(opt.prev_epoch_dir, file)
        prev = torch.load(last_data_path)

        # Bounding Box
        maskBB = np.array(
            [[600, 1900, 4700, 1700], [800, 1800, 4700, 1700]]).astype('float32')

        # Subdivide image in tiles and run a loop
        useTile = True
        writeTile2Disk = False
        do_viz_tile = True
        rowDivisor = 2
        colDivisor = 4

        print('Subdivinding image in tiles...')
        timerTile = AverageTimer(newline=True)
        tiles0, limits0 = subdivideImage(image0, rowDivisor=rowDivisor, colDivisor=colDivisor, overlap=0,
                                          viz=False, out_dir=opt.input_dir+'tiles0', writeTile2Disk=writeTile2Disk)
        tiles1, limits1 = subdivideImage(image1, rowDivisor=rowDivisor, colDivisor=colDivisor, overlap=0,
                                          viz=False, out_dir=opt.input_dir+'tiles1', writeTile2Disk=writeTile2Disk)
        timer.update('create_tiles')

        kpts0_full.append(
            np.full(np.shape(prev['keypoints0'][0].cpu().numpy()), -1, dtype=(float)))
        kpts1_full.append(
            np.full(np.shape(prev['keypoints0'][0].cpu().numpy()), -1, dtype=(float)))
        wasMatched.append(
            np.full((len(prev['keypoints0'][0].cpu().numpy())), -1, dtype=(float)))
        # mconf_full.append(
            # np.full((len(prev['keypoints0'][0].cpu().numpy())), 0, dtype=(float)))
       
        for t, tile0 in enumerate(tiles0):
            # Shift back to previuos keypoints
            def rectContains(rect, pt):
                logic = rect[0] < pt[0] < rect[2] and rect[1] < pt[1] < rect[3]
                return logic

            # prevTile = {k: v[0].cpu().numpy() for k, v in prev.items()}

            # Subract coordinates bounding box
            prev = torch.load(last_data_path)
            tmp = {k: v[0].cpu().numpy() for k, v in prev.items()}
            kpts0 = tmp['keypoints0'] - \
                np.array(maskBB[0][0:2]).astype('float32')

            ptsInTile = np.zeros(len(kpts0), dtype=(bool))
            for i, kk in enumerate(kpts0):
                ptsInTile[i] = rectContains(limits0[t], kk)
                ptsInTileIdx = np.where(ptsInTile == True)
            kpts0 = kpts0[ptsInTile] - \
                np.array(limits0[t][0:2]).astype('float32')

            # Build Prev tensor
            prevTile = {'keypoints0': [], 'scores0': [], 'descriptors0': []}
            prevTile['keypoints0'] = kpts0
            prevTile['scores0'] = tmp['scores0'][ptsInTile]
            prevTile['descriptors0'] = tmp['descriptors0'][:, ptsInTile]
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
            descriptors1 = descriptors1[:, matches0[valid]]
            scores1 = scores1[matches0[valid]]
            mconf = conf[valid]

            for i, pt in enumerate(kpts0):
                if predTile['matches0'][i] > -1:
                    wasMatched[cam][ptsInTileIdx[0][i]] = 1
                    kpts0_full[cam][ptsInTileIdx[0][i], :] = pt + \
                        np.array(limits0[t][0:2]).astype('float32')
                    kpts1_full[cam][ptsInTileIdx[0][i], :] = kpts1[predTile['matches0']
                        [i]] + np.array(limits1[t][0:2]).astype('float32')
                    # mconf_full[cam][ptsInTileIdx[0][i], :] = predTile['matches0']
                    #     [i]] + np.array(limits1[t][0:2]).astype('float32')
            # AGGIUNGERE DESCRITTORI!!!!

            if t < 1:
            #     # kpts0_full = kpts0.copy()
            #     # ptsInTileIdx_full = ptsInTileIdx[0].copy()
            #     # kpts1_full = kpts1.copy()
            #     # matches0_full = matches0.copy()
            #     mkpts0_full = mkpts0.copy()
            #     mkpts1_full = mkpts1.copy()
            #     descriptors1_full = descriptors1.copy()
            #     scores1_full = scores1.copy()
                mconf_full = mconf.copy()
            else:
            #     # kpts0_full = np.append(kpts0_full, kpts0, axis=0)
            #     # ptsInTileIdx_full = np.append(ptsInTileIdx_full, ptsInTileIdx[0], axis=0)
            #     # kpts1_full = np.append(kpts1_full, kpts1, axis=0)
            #     # matches0_full = np.append(matches0_full, matches0, axis=0)
            #     mkpts0_full = np.append(
            #         mkpts0_full, mkpts0 + np.array(limits0[t][0:2]).astype('float32'), axis=0)
            #     mkpts1_full = np.append(
            #         mkpts1_full, mkpts1 + np.array(limits1[t][0:2]).astype('float32'), axis=0)
            #     scores1_full = np.append(scores1_full, scores1, axis=0)
                mconf_full = np.append(mconf_full, mconf, axis=0)
            #     descriptors1_full = np.append(
            #         descriptors1_full, descriptors1, axis=1)

            if do_viz_tile:
                # Visualize the matches.
                vizTile_path= output_dir / '{}_{}_matches_tile{}.{}'.format(stem0, stem1, t, opt.viz_extension)
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
                    opt.fast_viz, opt.opencv_display, 'Matches', small_text)

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
                color, text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')


        timer.print('Finished pair {:5} of {:5}'.format(cam+1, len(pairs)))


    # %%

    # Retrieve points that were matched in both the images
    validTracked= [m == 2 for m in wasMatched[0] + wasMatched[1]]
    mkpts1_cam0 = kpts1_full[0][validTracked]
    mkpts1_cam1 = kpts1_full[1][validTracked]


    # Restore original image coordinates (not cropped) and run PyDegensac
    maskBB= np.array([[600, 1900, 4700, 1700], [800, 1800, 4700, 1700]]).astype('float32')
    mkpts1_cam0= mkpts1_cam0 + maskBB[0][0:2]
    mkpts1_cam1= mkpts1_cam1 + maskBB[1][0:2]

    F, inlMask= pydegensac.findFundamentalMatrix(mkpts1_cam0, mkpts1_cam1, px_th=3, conf=0.9,
                                                  max_iters=100000, laf_consistensy_coef=-1.0, error_type='sampson',
                                                  symmetric_error_check=True, enable_degeneracy_check=True)
    print('pydegensac found {} inliers ({:.2f}%)'.format(int(deepcopy(inlMask).astype(np.float32).sum()),
                    int(deepcopy(inlMask).astype(np.float32).sum())*100 / len(mkpts1_cam0)))
    mkpts1_cam0= mkpts1_cam0[inlMask]
    mkpts1_cam1= mkpts1_cam1[inlMask]
    timer.update('PyDegensac')

    # Viz point mached on both the images
    name0= pairs[0][1]
    name1= pairs[1][1]
    stem0, stem1= Path(name0).stem, Path(name1).stem
    viz_path= output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
    matches_path= output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
    image0, _, scales0= read_image(
        input_dir / name0, device, opt.resize, rot0, opt.resize_float, opt.equalize_hist)
    image1, _, scales1= read_image(
        input_dir / name1, device, opt.resize, rot1, opt.resize_float, opt.equalize_hist)

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
            text, viz_path, opt.show_keypoints,
            opt.fast_viz, opt.opencv_display, 'Matches', small_text)

        timer.update('viz_match')

    # Write to Disk
    out_matches= {'mkpts0': mkpts1_cam0, 'mkpts1': mkpts1_cam1,
                'match_confidence': []}
    np.savez(str(matches_path), **out_matches)
