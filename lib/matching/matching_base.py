import numpy as np
import cv2
import json
import pickle
import pydegensac

from easydict import EasyDict as edict
from typing import List
from collections import abc
from pathlib import Path

from lib.classes import Features
from lib.matching.match_pairs import match_pair
from lib.matching.track_matches import track_matches
from lib.utils import create_directory

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
        images: dict,
        features: dict,
    ) -> dict:

        cams = cfg.paths.cam_names

        # Load matching and tracking configurations
        with open(cfg.matching_cfg) as f:
            opt_matching = edict(json.load(f))
        with open(cfg.tracking_cfg) as f:
            opt_tracking = edict(json.load(f))

        for cam in cams:
            features[cam] = []

        for epoch in cfg.proc.epoch_to_process:
            print(f"Processing epoch {epoch}...")

            # opt_matching = cfg.matching.copy()
            epochdir = Path(cfg.paths.resdir) / f"epoch_{epoch}"

            # -- Find Matches at current epoch --#
            print(f"Run Superglue to find matches at epoch {epoch}")
            opt_matching.output_dir = epochdir
            pair = [
                images[cams[0]].get_image_path(epoch),
                images[cams[1]].get_image_path(epoch),
            ]
            # Call matching function
            matchedPts, matchedDescriptors, matchedPtsScores = match_pair(
                pair, cfg.images.bbox, opt_matching
            )

            # Store matches in features structure
            for jj, cam in enumerate(cams):
                # Dict keys are the cameras names, internal list contain epoches
                features[cam].append(Features())
                features[cam][epoch].append_features(
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

                trackoutdir = epochdir / f"from_t{epoch-1}"
                opt_tracking["output_dir"] = trackoutdir
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
                    features[cams[0]][epoch - 1].get_features_as_dict(),
                    features[cams[1]][epoch - 1].get_features_as_dict(),
                ]
                # Call actual tracking function
                tracked_cam0, tracked_cam1 = track_matches(
                    pairs, cfg.images.bbox, prevs, opt_tracking
                )
                # @TODO: keep track of the epoch in which feature is matched
                # @TODO: Check bounding box in tracking
                # @TODO: clean tracking code

                # Store all matches in features structure
                features[cams[0]][epoch].append_features(tracked_cam0)
                features[cams[1]][epoch].append_features(tracked_cam1)

            # Run Pydegensac to estimate F matrix and reject outliers
            F, inlMask = pydegensac.findFundamentalMatrix(
                features[cams[0]][epoch].get_keypoints(),
                features[cams[1]][epoch].get_keypoints(),
                px_th=1.5,
                conf=0.99999,
                max_iters=10000,
                laf_consistensy_coef=-1.0,
                error_type="sampson",
                symmetric_error_check=True,
                enable_degeneracy_check=True,
            )
            print(
                f"Matching at epoch {epoch}: pydegensac found {inlMask.sum()} \
                inliers ({inlMask.sum()*100/len(features[cams[0]][epoch]):.2f}%)"
            )
            features[cams[0]][epoch].remove_outliers_features(inlMask)
            features[cams[1]][epoch].remove_outliers_features(inlMask)

            # Write matched points to disk
            im_stems = images[cams[0]].get_image_stem(epoch), images[
                cams[1]
            ].get_image_stem(epoch)
            for jj, cam in enumerate(cams):
                features[cam][epoch].save_as_txt(
                    epochdir / f"{im_stems[jj]}_mktps.txt")
            with open(epochdir / f"{im_stems[0]}_{im_stems[1]}_features.pickle", "wb") as f:
                pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
            last_match_path = create_directory("res/last_epoch")
            with open(last_match_path / "last_features.pickle", "wb") as f:
                pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

            print("Matching completed")

        return features