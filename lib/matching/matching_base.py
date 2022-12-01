import numpy as np
import cv2
import json
import pickle
import pydegensac

from easydict import EasyDict as edict
from typing import List
from collections import abc
from pathlib import Path

from lib.base_classes.features import Features
from lib.matching.match_pairs import match_pair
from lib.matching.track_matches import track_matches
from lib.utils.utils import create_directory

# class MatchingBase(abc):

#     def match(self):
#         pass


# class MatchingTracking:

# def __init__(
#     self,
#     cfg: edict,
#     images: dict,
#     features: dict,
# ) -> dict:


def MatchingAndTracking(
    cfg: edict,
    epoch: int,
    images: dict,
    features: dict,
    epoch_dict: dict,
    # res_dir: Path,
    # prev_epoch_dir: Path,
) -> dict:

    # for epoch in cfg.proc.epoch_to_process:
    # print(f"Processing epoch {epoch}...")

    # epochdir = Path(cfg.paths.results_dir) / f"epoch_{epoch}/matching"

    epochdir = Path(cfg.paths.results_dir) / f"{epoch_dict[epoch]}/matching"
    cams = cfg.paths.camera_names

    # -- Find Matches at current epoch --#
    print(f"Run Superglue to find matches at epoch {epoch}")
    cfg.matching.output_dir = epochdir
    pair = [
        images[cams[0]].get_image_path(epoch),
        images[cams[1]].get_image_path(epoch),
    ]
    # Call matching function
    matchedPts, matchedDescriptors, matchedPtsScores = match_pair(
        pair, cfg.images.mask_bounding_box, cfg.matching
    )

    # Store matches in features structure
    for jj, cam in enumerate(cams):
        # Dict keys are the cameras names, internal list contain epoches
        features[epoch][cam] = Features()
        features[epoch][cam].append_features(
            {
                "kpts": matchedPts[jj],
                "descr": matchedDescriptors[jj],
                "score": matchedPtsScores[jj],
            }
        )
        # @TODO: Store match confidence!

    # === Track previous matches at current epoch ===#
    if cfg.proc.do_tracking and epoch > 0:
        print(f"Track points from epoch {epoch-1} to epoch {epoch}")

        # trackoutdir = epochdir / f"from_t{epoch-1}"
        trackoutdir = epochdir / f"from_{epoch_dict[epoch-1]}"

        cfg.tracking["output_dir"] = trackoutdir
        pairs = [
            [
                images[cams[0]].get_image_path(epoch - 1),
                images[cams[0]].get_image_path(epoch),
            ],
            [
                images[cams[1]].get_image_path(epoch - 1),
                images[cams[1]].get_image_path(epoch),
            ],
        ]
        prevs = [
            features[epoch - 1][cams[0]].get_features_as_dict(),
            features[epoch - 1][cams[1]].get_features_as_dict(),
        ]
        # Call actual tracking function
        tracked_cam0, tracked_cam1 = track_matches(
            pairs, cfg.images.mask_bounding_box, prevs, cfg.tracking
        )
        # @TODO: keep track of the epoch in which feature is matched
        # @TODO: Check bounding box in tracking
        # @TODO: clean tracking code

        # Store all matches in features structure
        features[epoch][cams[0]].append_features(tracked_cam0)
        features[epoch][cams[1]].append_features(tracked_cam1)

    # Run Pydegensac to estimate F matrix and reject outliers
    F, inlMask = pydegensac.findFundamentalMatrix(
        features[epoch][cams[0]].get_keypoints(),
        features[epoch][cams[1]].get_keypoints(),
        px_th=1.0,
        conf=0.99999,
        max_iters=10000,
        laf_consistensy_coef=-1.0,
        error_type="sampson",
        symmetric_error_check=True,
        enable_degeneracy_check=True,
    )
    print(
        f"Matching at epoch {epoch}: pydegensac found {inlMask.sum()} \
            inliers ({inlMask.sum()*100/len(features[epoch][cams[0]]):.2f}%)"
    )
    features[epoch][cams[0]].remove_outliers_features(inlMask)
    features[epoch][cams[1]].remove_outliers_features(inlMask)

    # Write matched points to disk
    im_stems = images[cams[0]].get_image_stem(epoch), images[cams[1]].get_image_stem(
        epoch
    )
    for jj, cam in enumerate(cams):
        features[epoch][cam].save_as_txt(epochdir / f"{im_stems[jj]}_mktps.txt")

    # Save current epoch features as pickle file
    fname = epochdir / f"{im_stems[0]}_{im_stems[1]}_features.pickle"
    with open(fname, "wb") as f:
        # keys = list(features.keys())
        # feat_epoch = {
        #     keys[0]: features[keys[0]][epoch],
        #     keys[1]: features[keys[1]][epoch],
        # }
        # pickle.dump(feat_epoch, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(features[epoch], f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save all features structure in last_epoch folder to resume the process
    # last_match_path = create_directory("res/last_epoch")
    # with open(last_match_path / "last_features.pickle", "wb") as f:
    #     pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Matching completed")

    return features
