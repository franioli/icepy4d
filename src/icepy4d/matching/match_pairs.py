import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import logging

from pathlib import Path

from icepy4d.utils import AverageTimer

from ..thirdparty.SuperGluePretrainedNetwork.models.matching import Matching
from ..thirdparty.SuperGluePretrainedNetwork.models.utils import (
    make_matching_plot,
    frame2tensor,
    process_resize,
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

# @TODO: This function is a duplicate of the one in track_matches!!!
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


def vizTileRes(viz_path, pred, image0, image1, matching, timer, opt):

    kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
    matches0 = pred["matches0"]
    conf = pred["matching_scores0"]

    # Keep the matching keypoints and descriptors.
    valid = matches0 > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches0[valid]]
    mconf = conf[valid]

    # Visualize the matches.
    color = cm.jet(mconf)
    text = [
        "SuperGlue",
        "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
        "Matches: {}".format(len(mkpts0)),
    ]

    # Display extra parameter info.
    k_thresh = matching.superpoint.config["keypoint_threshold"]
    m_thresh = matching.superglue.config["match_threshold"]
    small_text = [
        "Keypoint Threshold: {:.4f}".format(k_thresh),
        "Match Threshold: {:.2f}".format(m_thresh),
        "Image Pair: {}:{}".format(opt["imstem0"], opt["imstem1"]),
    ]

    make_matching_plot(
        image0,
        image1,
        kpts0,
        kpts1,
        mkpts0,
        mkpts1,
        color,
        text,
        viz_path,
        opt["show_keypoints"],
        opt["fast_viz"],
        opt["opencv_display"],
        "Matches",
        small_text,
    )

    timer.update("viz_match")


def match_pair(pair, maskBB, opt):

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
    assert not (
        opt.fast_viz and not opt.viz_matches
    ), "Must use --viz_matches with --fast_viz"
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
            "weights": opt.weights,
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

    timer = AverageTimer()
    name0, name1 = pair[:2]
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    matches_path = output_dir / "{}_{}_matches.npz".format(stem0, stem1)
    viz_path = output_dir / "{}_{}_matches.{}".format(stem0, stem1, opt.viz_extension)

    rot0, rot1 = 0, 0
    image0, inp0, scales0 = read_image(
        name0, device, opt.resize, rot0, opt.resize_float, maskBB[0]
    )
    image1, inp1, scales1 = read_image(
        name1, device, opt.resize, rot1, opt.resize_float, maskBB[1]
    )
    if image0 is None or image1 is None:
        logging.error("Problem reading image pair: {} {}".format(name0, name1))
        exit(1)
    timer.update("load_image")

    if opt.useTile:

        # Subdivide image in tiles and run a loop
        tiles0, limits0 = generateTiles(
            image0,
            rowDivisor=opt.rowDivisor,
            colDivisor=opt.colDivisor,
            overlap=opt.overlap,
            viz=opt.do_viz_tile,
            out_dir=output_dir / "tiles0",
            writeTile2Disk=opt.writeTile2Disk,
        )
        tiles1, limits1 = generateTiles(
            image1,
            rowDivisor=opt.rowDivisor,
            colDivisor=opt.colDivisor,
            overlap=opt.overlap,
            viz=opt.do_viz_tile,
            out_dir=output_dir / "tiles1",
            writeTile2Disk=opt.writeTile2Disk,
        )
        logging.info(f"Images subdivided in {opt.rowDivisor}x{opt.colDivisor} tiles")
        timer.update("create_tiles")

        timerTile = AverageTimer()
        for t0, tile0 in enumerate(tiles0):
            # for t1, tile1 in enumerate(tiles1):
            # Perform the matching.
            t1 = t0
            inp0 = frame2tensor(tiles0[t0], device)
            inp1 = frame2tensor(tiles1[t1], device)
            predTile = matching({"image0": inp0, "image1": inp1})
            predTile = {k: v[0].cpu().numpy() for k, v in predTile.items()}
            timerTile.update("matcher")

            kpts0, kpts1 = predTile["keypoints0"], predTile["keypoints1"]
            descriptors0, descriptors1 = (
                predTile["descriptors0"],
                predTile["descriptors1"],
            )
            scores0, scores1 = predTile["scores0"], predTile["scores1"]
            matches0, matches1 = (
                predTile["matches0"],
                predTile["matches1"],
            )
            conf = predTile["matching_scores0"]
            valid = matches0 > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches0[valid]]
            descriptors0 = descriptors0[:, valid]
            descriptors1 = descriptors1[:, matches0[valid]]
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
                mkpts0_full = np.append(
                    mkpts0_full,
                    mkpts0 + np.array(limits0[t0][0:2]).astype("float32"),
                    axis=0,
                )
                mkpts1_full = np.append(
                    mkpts1_full,
                    mkpts1 + np.array(limits1[t1][0:2]).astype("float32"),
                    axis=0,
                )
                scores0_full = np.append(scores0_full, scores0, axis=0)
                scores1_full = np.append(scores1_full, scores1, axis=0)
                conf_full = np.append(conf_full, conf, axis=0)
                descriptors0_full = np.append(descriptors0_full, descriptors0, axis=1)
                descriptors1_full = np.append(descriptors1_full, descriptors1, axis=1)

            if opt.do_viz_tile:
                vizTile_path = output_dir / "{}_{}_matches_tile{}_{}.{}".format(
                    stem0, stem1, t0, t1, opt.viz_extension
                )
                tile_print_opt = {
                    "imstem0": stem0 + "_" + str(t0),
                    "imstem1": stem1 + "_" + str(t1),
                    "show_keypoints": True,
                    "fast_viz": opt.fast_viz,
                    "opencv_display": opt.opencv_display,
                }
                vizTileRes(
                    vizTile_path,
                    predTile,
                    tiles0[t0],
                    tiles1[t1],
                    matching,
                    timerTile,
                    tile_print_opt,
                )

            timerTile.print(
                "Finished Tile Pairs {:2} - {:2} of {:2}".format(t0, t1, len(tiles0))
            )

    # Restore original image coordinates (not cropped) and run PyDegensac
    mkpts0_full = mkpts0_full + np.array(maskBB[0][0:2]).astype("float32")
    mkpts1_full = mkpts1_full + np.array(maskBB[1][0:2]).astype("float32")

    mconf = conf_full
    mkpts0 = mkpts0_full
    mkpts1 = mkpts1_full
    scores0 = scores0_full
    scores1 = scores1_full
    descriptors0 = descriptors0_full
    descriptors1 = descriptors1_full

    # Write the matches to disk.
    out_matches = {"mkpts0": mkpts0, "mkpts1": mkpts1, "match_confidence": mconf}
    np.savez(str(matches_path), **out_matches)

    # Write tensors to disk
    prev0 = {}
    prev0["keypoints0"] = [torch.from_numpy(mkpts0).to(device)]
    prev0["scores0"] = (torch.from_numpy(scores0).to(device),)
    prev0["descriptors0"] = [torch.from_numpy(descriptors0).to(device)]
    # torch.save(prev0, str(matches_path.parent / '{}_tensor_0.pt'.format(matches_path.stem)))

    prev1 = {}
    prev1["keypoints0"] = [torch.from_numpy(mkpts1).to(device)]
    prev1["scores0"] = (torch.from_numpy(scores1).to(device),)
    prev1["descriptors0"] = [torch.from_numpy(descriptors1).to(device)]
    # torch.save(prev1, str(matches_path.parent / '{}_tensor_1.pt'.format(matches_path.stem)))

    if opt.viz_matches:
        # Visualize the matches.
        color = cm.jet(mconf)
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
            "Matches: {}".format(len(mkpts0)),
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
            mkpts0 - maskBB[0][0:2].astype("float32"),
            mkpts1 - maskBB[1][0:2].astype("float32"),
            mkpts0 - maskBB[0][0:2].astype("float32"),
            mkpts1 - maskBB[1][0:2].astype("float32"),
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

    timer.print("Finished pair")

    # Free cuda memory and return variables
    torch.cuda.empty_cache()

    # Build output dict
    out_matches = {"mkpts0": mkpts0, "mkpts1": mkpts1, "match_confidence": mconf}
    np.savez(str(matches_path), **out_matches)

    return [mkpts0, mkpts1], [descriptors0, descriptors1], [scores0, scores1]


# if __name__ == '__main__':
#     import os, json

#     epoch = 0
#     matching_config = 'config/opt_matching.json'
#     maskBB = [[400,1500,5500,4000], [600,1400,5700,3900]]
#     epochdir = os.path.join('res','epoch_'+str(epoch))
#     with open(matching_config,) as f:
#         opt_matching = json.load(f)
#     opt_matching['output_dir'] = epochdir
#     pair = [images[0][epoch], images[1][epoch]]
#     maskBB = np.array(maskBB).astype('int')
#     matchedPts, matchedDescriptors, matchedPtsScores = match_pair(pair, maskBB, opt_matching)
