import pickle
import pydegensac
import logging

from easydict import EasyDict as edict
from typing import List
from pathlib import Path

from .match_pairs import match_pair
from .track_matches import track_matches

from ..base_classes.features import Features


def MatchingAndTracking(
    cfg: edict,
    epoch: int,
    images: dict,
    features: dict,
    epoch_dict: dict,
) -> dict:

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

        # If features from previous epoch are not already present in features object (e.g. when the process started from and epoch different that 0), try to load them from disk. If it fails, skip tracking.
        if epoch - 1 not in features.keys():
            path = Path(cfg.paths.results_dir) / f"{epoch_dict[epoch-1]}/matching"
            logging.warning(
                f"Feature from previous epoch not available in Features object. Try to load it from disk at {path}"
            )
            try:
                fname = list(path.glob("*.pickle"))
                try:
                    with open(fname[0], "rb") as f:
                        loaded_features = pickle.load(f)
                except:
                    raise FileNotFoundError(f"Invalid pickle file in {path}.")
                features[epoch - 1] = loaded_features
            except FileNotFoundError as err:
                logging.error(err)

        if epoch - 1 in features.keys():
            prevs = [features[epoch - 1][cam].get_features_as_dict() for cam in cams]

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
        else:
            logging.warning(
                f"Skipping tracking from epoch {epoch_dict[epoch-1]} to {epoch_dict[epoch]}"
            )

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
