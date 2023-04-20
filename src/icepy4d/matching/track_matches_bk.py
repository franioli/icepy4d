from pathlib import Path
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import logging

from ..thirdparty.SuperGluePretrainedNetwork.models.matching import Matching
from ..thirdparty.SuperGluePretrainedNetwork.models.utils import (
    make_matching_plot,
    AverageTimer,
    process_resize,
    frame2tensor,
)
from ..tiles import generateTiles


torch.set_grad_enabled(False)

# SuperPoint Parameters
NMS_RADIUS = 3

# SuperGlue Parameters
SINKHORN_ITERATIONS = 100

# Processing parameters
RESIZE_FLOAT = True
VIZ_EXTENSION = "png"
OPENCV_DISPLAY = False
SHOW_KEYPOINTS = False
CACHE = False


# @TODO: This function is a duplicate of the one in match_pairs!!!
# It is a replacement of the SuperGlue one because of the different input parametets.
# This must be fixed! Only ONE read_image function must exist!
# (There is also read_image function implemented from scratch in icepy4d)
def read_image(path, device, resize=-1, rotation=0, resize_float=True, crop=[]):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype("float32"), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype("float32")

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]
    if np.any(crop):
        image = image[crop[1] : crop[3], crop[0] : crop[2]]

    inp = frame2tensor(image, device)
    return image, inp, scales


def track_matches(pairs, maskBB, prevs, track_id, opt):

    opt.resize_float = RESIZE_FLOAT
    opt.viz_extension = VIZ_EXTENSION
    opt.opencv_display = OPENCV_DISPLAY
    opt.show_keypoints = SHOW_KEYPOINTS
    opt.cache = CACHE

    assert not (
        opt.opencv_display and not opt.viz_matches
    ), "Must use --viz with --opencv_display"
    assert not (
        opt.opencv_display and not opt.fast_viz
    ), "Cannot use --opencv_display without --fast_viz"
    assert not (opt.fast_viz and not opt.viz_matches), "Must use --viz with --fast_viz"
    assert not (
        opt.fast_viz and opt.viz_extension == "pdf"
    ), "Cannot use pdf extension with --fast_viz"

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        logging.info(f"Will resize to {opt.resize[0]}x{opt.resize[1]} (WxH)")
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        logging.info(f"Will resize max dimension to {opt.resize[0]}")
    elif len(opt.resize) == 1:
        logging.info(f"Will not resize images")
    else:
        raise ValueError("Cannot specify more than two integers for --resize")

    # Load the SuperPoint and SuperGlue models.
    device = "cuda" if torch.cuda.is_available() and not opt.force_cpu else "cpu"
    logging.info(f"Running inference on device {device}")
    config = {
        "superpoint": {
            "nms_radius": NMS_RADIUS,
            "keypoint_threshold": opt.keypoint_threshold,
            "max_keypoints": opt.max_keypoints,
        },
        "superglue": {
            "weights": opt.superglue,
            "sinkhorn_iterations": SINKHORN_ITERATIONS,
            "match_threshold": opt.match_threshold,
        },
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Will write matches to directory {output_dir}")
    if opt.viz_matches:
        logging.info(f"Will write visualization images to directory {output_dir}")

    # Inizialize lists for storing matching points
    kpts0_full = []
    kpts1_full = []
    wasMatched = []
    # mconf_full = []
    descriptors1_full = []
    scores1_full = []
    track_id1_full = []

    timer = AverageTimer()

    #%% Run tracking
    for cam, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / "{}_{}_matches.npz".format(stem0, stem1)
        viz_path = output_dir / "{}_{}_matches.{}".format(
            stem0, stem1, opt["viz_extension"]
        )

        # Load the image pair.
        rot0, rot1 = 0, 0
        image0, inp0, scales0 = read_image(
            name0,
            device,
            opt["resize"],
            rot0,
            opt["resize_float"],
            maskBB[0],
        )
        image1, inp1, scales1 = read_image(
            name1,
            device,
            opt["resize"],
            rot1,
            opt["resize_float"],
            maskBB[1],
        )
        if image0 is None or image1 is None:
            logging.error("Problem reading image pair: {} {}".format(name0, name1))
            exit(1)
        timer.update("load_image")

        # Subdivide image in tiles
        do_viz = opt.viz_matches
        useTile = opt.useTile
        writeTile2Disk = opt.writeTile2Disk
        do_viz_tile = opt.do_viz_tile
        rowDivisor = opt.rowDivisor
        colDivisor = opt.colDivisor

        timerTile = AverageTimer()
        tiles0, limits0 = generateTiles(
            image0,
            rowDivisor=rowDivisor,
            colDivisor=colDivisor,
            overlap=0,
            viz=do_viz_tile,
            out_dir=output_dir / "tiles0",
            writeTile2Disk=writeTile2Disk,
        )
        tiles1, limits1 = generateTiles(
            image1,
            rowDivisor=rowDivisor,
            colDivisor=colDivisor,
            overlap=0,
            viz=do_viz_tile,
            out_dir=output_dir / "tiles1",
            writeTile2Disk=writeTile2Disk,
        )
        logging.info(f"Images subdivided in {rowDivisor}x{colDivisor} tiles")
        timer.update("create_tiles")

        # import pdb
        # pdb.set_trace()

        # try:
        #     if torch.is_tensor(prevs[cam]['keypoints0'][0]):
        #         prev = {k: v[0].cpu().numpy() for k, v in prevs[cam].items()}
        #     logging.info('Prev data are tensors: converted to np')
        # except:
        prev = prevs[cam]

        kpts0_full.append(np.full(np.shape(prev["keypoints0"]), -1, dtype=(float)))
        kpts1_full.append(np.full(np.shape(prev["keypoints0"]), -1, dtype=(float)))
        wasMatched.append(np.full(len(prev["keypoints0"]), -1, dtype=(float)))
        descriptors1_full.append(
            np.full(np.shape(prev["descriptors0"]), -1, dtype=(float))
        )
        scores1_full.append(np.full(len(prev["scores0"]), -1, dtype=(float)))
        # mconf_full.append(np.full((len(prev['keypoints0'])), 0, dtype=(float)))
        track_id1_full.append(np.full(len(track_id), -1, dtype=(int)))

        # TODO: CHECK IT!
        # Subract coordinates bounding box
        kpts0 = prev["keypoints0"] - np.array(maskBB[cam][0:2]).astype("float32")

        # import cv2
        # ttt = 3
        # pts0 = kpts0 - np.array(limits0[ttt][0:2]).astype('float32')
        # img0 = np.uint8(tiles0[ttt])
        # img0_kpts = cv2.drawKeypoints(img0,cv2.KeyPoint.convert(pts0),img0,(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt.imshow(img0_kpts)
        # plt.show()

        for t, tile0 in enumerate(tiles0):
            # Shift back to previuos keypoints
            def rectContains(rect, pt):
                logic = rect[0] < pt[0] < rect[2] and rect[1] < pt[1] < rect[3]
                return logic

            # Keep only kpts in current tile
            ptsInTile = np.zeros(len(kpts0), dtype=(bool))
            for i, kk in enumerate(kpts0):
                ptsInTile[i] = rectContains(limits0[t], kk)
            ptsInTileIdx = np.where(ptsInTile == True)[0]
            kpts0_tile = kpts0[ptsInTile] - np.array(limits0[t][0:2]).astype("float32")
            track_id0_tile = np.array(track_id)[ptsInTile]

            # Build Prev tensor
            prevTile = {
                "keypoints0": kpts0_tile,
                "scores0": prev["scores0"][ptsInTile],
                "descriptors0": prev["descriptors0"][:, ptsInTile],
            }

            # Perform the matching.
            inp0 = frame2tensor(tiles0[t], device)
            inp1 = frame2tensor(tiles1[t], device)
            prevTile = {
                k: [torch.from_numpy(v).to(device)] for k, v in prevTile.items()
            }
            prevTile["image0"] = inp0
            predTile = matching({**prevTile, "image1": inp1})
            predTile = {k: v[0].cpu().numpy() for k, v in predTile.items()}
            timerTile.update("matcher")

            # Retrieve points
            kpts1 = predTile["keypoints1"]
            descriptors1 = predTile["descriptors1"]
            scores1 = predTile["scores1"]
            matches0 = predTile["matches0"]
            conf = predTile["matching_scores0"]
            valid = matches0 > -1
            mkpts0 = kpts0_tile[valid]
            mkpts1 = kpts1[matches0[valid]]
            mconf = conf[valid]
            track_id1 = track_id0_tile[valid]

            # pts0 = mkpts0
            # img0 = np.uint8(tiles0[t])
            # img0_kpts = cv2.drawKeypoints(img0,cv2.KeyPoint.convert(pts0),img0,(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # plt.imshow(img0_kpts)
            # plt.show()

            for i, pt in enumerate(kpts0_tile):
                if predTile["matches0"][i] > -1:
                    wasMatched[cam][ptsInTileIdx[i]] = 1
                    kpts0_full[cam][ptsInTileIdx[i], :] = pt + np.array(
                        limits0[t][0:2]
                    ).astype("float32")
                    kpts1_full[cam][ptsInTileIdx[i], :] = kpts1[
                        predTile["matches0"][i], :
                    ] + np.array(limits1[t][0:2]).astype("float32")
                    descriptors1_full[cam][:, ptsInTileIdx[i]] = descriptors1[
                        :, predTile["matches0"][i]
                    ]
                    scores1_full[cam][ptsInTileIdx[i]] = scores1[
                        predTile["matches0"][i]
                    ]
                    track_id1_full[cam][ptsInTileIdx[i]] = track_id0_tile[i]

                    # mconf_full[cam][ptsInTileIdx[0][i], :] = predTile['matches0']
                    #     [i]] + np.array(limits1[t][0:2]).astype('float32')
                    # TO DO: Keep track of the matching scores

            # pts0 =  kpts1_full[0][ptsInTileIdx, :] + maskBB[1][0:2].astype('float32')
            # img0 = cv2.cvtColor(images[0][1], cv2.COLOR_BGR2GRAY)
            # img0_kpts = cv2.drawKeypoints(img0,cv2.KeyPoint.convert(pts0),img0,(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imwrite('dummy_out.jpg', img0_kpts)

            if t < 1:
                mconf_full = mconf.copy()
            else:
                mconf_full = np.append(mconf_full, mconf, axis=0)

            if do_viz_tile:
                # Visualize the matches.
                vizTile_path = output_dir / "{}_{}_matches_tile{}.{}".format(
                    stem0, stem1, t, opt.viz_extension
                )
                color = cm.jet(mconf)
                text = [
                    "SuperGlue",
                    "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
                    "Matches: {}".format(len(mkpts0)),
                ]
                k_thresh = matching.superpoint.config["keypoint_threshold"]
                m_thresh = matching.superglue.config["match_threshold"]
                small_text = [
                    "Keypoint Threshold: {:.4f}".format(k_thresh),
                    "Match Threshold: {:.2f}".format(m_thresh),
                    "Image Pair: {}:{}".format(stem0, stem1),
                ]
                make_matching_plot(
                    tiles0[t],
                    tiles1[t],
                    kpts0,
                    kpts1,
                    mkpts0,
                    mkpts1,
                    color,
                    text,
                    vizTile_path,
                    True,
                    opt.fast_viz,
                    opt.opencv_display,
                    "Matches",
                    small_text,
                )

            # if do_viz_tile:
            #     predTile['keypoints0']= kpts0
            #     vizTile_path= output_dir / '{}_{}_matches_tile{}.{}'.format(stem0, stem1, t, opt.viz_extension)
            #     tile_logging.info_opt= {'imstem0': stem0+'_'+str(t), 'imstem1': stem1+'_'+str(t), 'show_keypoints': True,
            #            'fast_viz': opt.fast_viz, 'opencv_display': opt.opencv_display}
            #     vizTileRes(vizTile_path, predTile,
            #                tiles0[t], tiles1[t], matching, timerTile, tile_logging.info_opt)

            timerTile.print(f"Finished Tile Pairs {t:2} of {len(tiles0):2}")

        if do_viz:
            # Visualize the matches.
            val = wasMatched[cam] == 1
            color = cm.jet(mconf_full)  # mconf_full
            text = [
                "SuperGlue",
                "Keypoints: {}:{}".format(len(kpts0_full[cam]), len(kpts1_full[cam])),
                "Matches: {}".format(len(kpts1_full[cam])),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append("Rotation: {}:{}".format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.superpoint.config["keypoint_threshold"]
            m_thresh = matching.superglue.config["match_threshold"]
            small_text = [
                "Keypoint Threshold: {:.4f}".format(k_thresh),
                "Match Threshold: {:.2f}".format(m_thresh),
                "Image Pair: {}:{}".format(stem0, stem1),
            ]

            make_matching_plot(
                image0,
                image1,
                kpts0_full[cam][val],
                kpts1_full[cam][val],
                kpts0_full[cam][val],
                kpts1_full[cam][val],
                color,
                text,
                viz_path,
                opt.show_keypoints,
                opt.fast_viz,
                opt.opencv_display,
                "Matches",
                small_text,
            )

            timer.update("viz_match")

        timer.print("Finished pair {:5} of {:5}".format(cam + 1, len(pairs)))

    # %%

    # Retrieve points that were matched in both the images
    validTracked = [m == 2 for m in wasMatched[0] + wasMatched[1]]
    mkpts1_cam0 = kpts1_full[0][validTracked]
    mkpts1_cam1 = kpts1_full[1][validTracked]
    descr1_cam0 = descriptors1_full[0][:, validTracked]
    descr1_cam1 = descriptors1_full[1][:, validTracked]
    scores1_cam0 = scores1_full[0][validTracked]
    scores1_cam1 = scores1_full[1][validTracked]

    track_id_cam0 = np.array(track_id)[validTracked].astype(np.int32)
    track_id_cam1 = track_id1_full[1][validTracked].astype(np.int32)

    # Restore original image coordinates (not cropped)
    mkpts1_cam0 = mkpts1_cam0 + np.array(maskBB[1][0:2]).astype("float32")
    mkpts1_cam1 = mkpts1_cam1 + np.array(maskBB[1][0:2]).astype("float32")

    # pts0 =  mkpts1_cam1
    # img0 = cv2.cvtColor(images[1][1], cv2.COLOR_BGR2GRAY)
    # img0_kpts = cv2.drawKeypoints(img0,cv2.KeyPoint.convert(pts0),img0,(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('dummy_out.jpg', img0_kpts)

    # Viz point mached on both the images
    name0 = pairs[0][1]
    name1 = pairs[1][1]
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    viz_path = output_dir / "{}_{}_matches.{}".format(stem0, stem1, opt.viz_extension)
    matches_path = output_dir / "{}_{}_matches.npz".format(stem0, stem1)
    image0, _, _ = read_image(
        name0,
        device,
        opt.resize,
        rot0,
        opt.resize_float,
        maskBB[0],
    )
    image1, _, _ = read_image(
        name1,
        device,
        opt.resize,
        rot0,
        opt.resize_float,
        maskBB[1],
    )

    if do_viz:
        # Visualize the matches.
        color = cm.jet(mconf_full)
        text = [
            "SuperGlue",
            # 'Keypoints: {}:{}'.format(len(kpts0_full[cam]), len(kpts1_full[cam])),
            "Matches: {}".format(len(mkpts1_cam1)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append("Rotation: {}:{}".format(rot0, rot1))

        # Display extra parameter info.
        k_thresh = matching.superpoint.config["keypoint_threshold"]
        m_thresh = matching.superglue.config["match_threshold"]
        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
            "Image Pair: {}:{}".format(stem0, stem1),
        ]

        make_matching_plot(
            image0,
            image1,
            mkpts1_cam0 - maskBB[1][0:2],
            mkpts1_cam1 - maskBB[1][0:2],
            mkpts1_cam0 - maskBB[1][0:2],
            mkpts1_cam1 - maskBB[1][0:2],
            color,
            text,
            viz_path,
            opt.show_keypoints,
            opt.fast_viz,
            opt.opencv_display,
            "Matches",
            small_text,
        )

        timer.update("viz_match")

    # Write to Disk
    out_matches = {"mkpts0": mkpts1_cam0, "mkpts1": mkpts1_cam1, "match_confidence": []}
    np.savez(str(matches_path), **out_matches)

    # Free cuda memory and return variables
    torch.cuda.empty_cache()

    tracked_cam0 = {
        "kpts": mkpts1_cam0,
        "descr": descr1_cam0,
        "score": scores1_cam0,
        "track_id": track_id_cam0,
    }
    tracked_cam1 = {
        "kpts": mkpts1_cam1,
        "descr": descr1_cam1,
        "score": scores1_cam1,
        "track_id": track_id_cam1,
    }

    return tracked_cam0, tracked_cam1


if __name__ == "__main__":
    import pickle

    with open("dummy.pickle", "rb") as f:
        tmp = pickle.load(f)

    features = tmp[0]
    pairs = tmp[1]
    maskBB = tmp[2]
    prevs = tmp[3]
    opt = tmp[4]

    tracked_cam0, tracked_cam1 = track_matches(pairs, maskBB, prevs, opt)
