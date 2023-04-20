import pickle
import pydegensac
import logging

from easydict import EasyDict as edict
from typing import List, Union
from pathlib import Path

# from .utils import load_matches_from_disk
from .match_pairs import match_pair
from .track_matches import track_matches

from ..classes.features import Features

DEBUG = True


def MatchingAndTracking(
    cfg: edict,
    epoch: int,
    images: dict,
    features: dict,
    epoch_dict: dict,
) -> dict:

    epochdir = Path(cfg.paths.results_dir) / f"{epoch_dict[epoch]}/matching"
    cams = cfg.paths.camera_names

    # === Track previous matches at current epoch ===#
    if cfg.proc.do_tracking and epoch > 0:
        logging.info(f"Track points from epoch {epoch-1} to epoch {epoch}")

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

            cam = cams[0]
            prev_track_id = features[epoch - 1][cam].get_track_ids()

            # Call actual tracking function
            tracked_cam0, tracked_cam1 = track_matches(
                pairs, cfg.images.mask_bounding_box, prevs, prev_track_id, cfg.tracking
            )
            tracked_kpts_idx = list(tracked_cam1["track_id"])

            # @TODO: keep track of the epoch in which feature is matched
            # @TODO: Check bounding box in tracking
            # @TODO: clean tracking code

            # Store all matches in features structure
            features[epoch][cams[0]].append_features_from_numpy(
                tracked_cam0["kpts"][:, 0:1],
                tracked_cam0["kpts"][:, 1:2],
                tracked_cam0["descr"],
                tracked_cam0["score"],
                track_ids=tracked_kpts_idx,
                epoch=epoch - 1,
            )
            features[epoch][cams[1]].append_features_from_numpy(
                tracked_cam1["kpts"][:, 0:1],
                tracked_cam1["kpts"][:, 1:2],
                tracked_cam1["descr"],
                tracked_cam1["score"],
                track_ids=tracked_kpts_idx,
                epoch=epoch - 1,
            )

            # For debugging
            if DEBUG:
                # from ..visualization.visualization import make_matching_plot

                f_list = {
                    epoch: features[epoch][cams[0]]
                    for epoch in range(cfg.proc.epoch_to_process[0], epoch + 1)
                }

                # make_matching_plot(
                #     images[cams[0]].read_image(epoch).value,
                #     images[cams[1]].read_image(epoch).value,
                #     features[epoch][cams[0]].kpts_to_numpy(),
                #     features[epoch][cams[1]].kpts_to_numpy(),
                #     path="ep181.png",
                # )
                # idx = list(tracked_cam1["track_id"])
                # aa = Features()
                # aa._values = features[epoch - 1][cams[0]].get_feature_by_index(idx)
                # bb = Features()
                # bb._values = features[epoch - 1][cams[1]].get_feature_by_index(idx)
                # make_matching_plot(
                #     images[cams[0]].read_image(epoch-1).value,
                #     images[cams[1]].read_image(epoch-1).value,
                #     aa.kpts_to_numpy(),
                #     bb.kpts_to_numpy(),
                #     path="ep180.png",
                # )

        else:
            logging.warning(
                f"Skipping tracking from epoch {epoch_dict[epoch-1]} to {epoch_dict[epoch]}"
            )

    # -- Find Matches at current epoch --#
    logging.info(f"Run Superglue to find matches at epoch {epoch}")
    cfg.matching.output_dir = epochdir
    pair = [
        images[cams[0]].get_image_path(epoch),
        images[cams[1]].get_image_path(epoch),
    ]
    # Call matching function
    matchedPts, matchedDescriptors, matchedPtsScores = match_pair(
        pair, cfg.images.mask_bounding_box, cfg.matching
    )

    # Store matches in Features structure
    for jj, cam in enumerate(cams):
        x = matchedPts[jj][:, 0:1]
        y = matchedPts[jj][:, 1:2]
        if epoch > cfg.proc.epoch_to_process[0]:
            last_track_id = features[epoch - 1][cam].get_track_ids()[-1]
            features[epoch][cam].set_last_track_id(last_track_id)
        features[epoch][cam].append_features_from_numpy(
            x,
            y,
            descr=matchedDescriptors[jj],
            scores=matchedPtsScores[jj],
            epoch=epoch,
        )
        # @TODO: Store match confidence!
    logging.info(f"SuperGlue found {len(features[epoch][cam])} matches")

    # For debugging
    f_list = {
        epoch: features[epoch][cams[0]]
        for epoch in range(cfg.proc.epoch_to_process[0], epoch + 1)
    }

    # Run Pydegensac to estimate F matrix and reject outliers
    logging.info(
        f"Geometric verification of the matches - Pydegensac parameters:  threshold {cfg.matching.pydegensac_threshold} [px], confidence: {cfg.matching.pydegensac_confidence}"
    )
    F, inlMask = pydegensac.findFundamentalMatrix(
        features[epoch][cams[0]].kpts_to_numpy(),
        features[epoch][cams[1]].kpts_to_numpy(),
        px_th=cfg.matching.pydegensac_threshold,
        conf=cfg.matching.pydegensac_confidence,
        max_iters=10000,
        laf_consistensy_coef=-1.0,
        error_type="sampson",
        symmetric_error_check=True,
        enable_degeneracy_check=True,
    )
    logging.info(
        f"Pydegensac found {inlMask.sum()} inliers ({inlMask.sum()*100/len(features[epoch][cams[0]]):.2f}%)"
    )

    features[epoch][cams[0]].filter_feature_by_mask(inlMask)
    features[epoch][cams[1]].filter_feature_by_mask(inlMask)

    # Write matched points to disk
    im_stems = (
        images[cams[0]].get_image_stem(epoch),
        images[cams[1]].get_image_stem(epoch),
    )
    for jj, cam in enumerate(cams):
        features[epoch][cam].save_as_txt(epochdir / f"{im_stems[jj]}_mktps.txt")

    # Save current epoch features as pickle file
    fname = epochdir / f"{im_stems[0]}_{im_stems[1]}_features.pickle"
    with open(fname, "wb") as f:
        pickle.dump(features[epoch], f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Matching completed")

    return features
