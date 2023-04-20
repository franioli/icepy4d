import pickle
import pydegensac
import logging

from easydict import EasyDict as edict
from typing import List, Union
from pathlib import Path

from .match_pairs import match_pair
from .track_matches import track_matches

import icepy4d.classes as icepy4d_classes
from icepy4d.classes.features import Features
from icepy4d.utils.timer import AverageTimer


def tracking_base(
    images: icepy4d_classes.ImagesDict,
    prev_features: icepy4d_classes.FeaturesDictEpoch,
    camera_names: List[str],
    epoch_dict: icepy4d_classes.EpochDict,
    epoch: int,
    cfg: edict,
    epoch_dir: Union[Path, str],
) -> icepy4d_classes.FeaturesDictEpoch:

    if not isinstance(cfg, dict):
        raise TypeError("opt must be a dictionary")
    required_keys = [
        "weights",
        "keypoint_threshold",
        "max_keypoints",
        "match_threshold",
        "force_cpu",
    ]
    missing_keys = [key for key in required_keys if key not in cfg]
    if missing_keys:
        raise KeyError(
            f"Missing required keys: {', '.join(missing_keys)} in SuperGlue Matcher option dictionary"
        )

    timer = AverageTimer()
    cams = camera_names
    epoch_dir = Path(epoch_dir)
    tracking_dir = epoch_dir / f"matching/from_{epoch_dict[epoch-1]}"
    tracking_dir.mkdir(exist_ok=True, parents=True)
    cfg["output_dir"] = tracking_dir
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

    # Try to load features from previous epoch from disk. If it fails, skip tracking.
    logging.info(
        f"Tracking features from epoch {epoch_dict[epoch-1]} to {epoch_dict[epoch]}"
    )
    prev_path = epoch_dir.parent / f"{epoch_dict[epoch-1]}/matching"
    fname = list(prev_path.glob("*.pickle"))
    try:
        with open(fname[0], "rb") as f:
            prev_features = pickle.load(f)
            logging.info("Previous features loaded.")

    except:
        raise FileNotFoundError(
            f"Invalid pickle file in {prev_path}. Skipping tracking from epoch {epoch_dict[epoch-1]} to {epoch_dict[epoch]}"
        )

    prevs = [prev_features[cam].get_features_as_dict() for cam in cams]
    prev_track_id = prev_features[cams[0]].get_track_ids()
    timer.update("previous feature loaded")

    # Call actual tracking function
    # @TODO: clean tracking code
    bbox = [[900, 1000, 5600, 3900], [200, 1100, 4900, 4000]]
    tracked_cam0, tracked_cam1 = track_matches(pairs, bbox, prevs, prev_track_id, cfg)
    tracked_kpts_idx = list(tracked_cam1["track_id"])
    timer.update("tracking")

    # Store all matches in features structure
    features = {cam: icepy4d_classes.Features() for cam in cams}
    features[cams[0]].append_features_from_numpy(
        tracked_cam0["kpts"][:, 0:1],
        tracked_cam0["kpts"][:, 1:2],
        tracked_cam0["descr"],
        tracked_cam0["score"],
        track_ids=tracked_kpts_idx,
        epoch=epoch - 1,
    )
    features[cams[1]].append_features_from_numpy(
        tracked_cam1["kpts"][:, 0:1],
        tracked_cam1["kpts"][:, 1:2],
        tracked_cam1["descr"],
        tracked_cam1["score"],
        track_ids=tracked_kpts_idx,
        epoch=epoch - 1,
    )
    timer.update("export")
    timer.print("tracking")

    return features
