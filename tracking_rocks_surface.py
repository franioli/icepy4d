#%%
import numpy as np
import logging

from pathlib import Path

# icepy classes
import src.icepy.classes as icepy_classes

# icepy libraries
import src.icepy.utils.initialization as initialization
import src.icepy.utils as icepy_utils

from src.icepy.matching.matching_base import (
    MatchingAndTracking,
    load_matches_from_disk,
)


if __name__ == "__main__":

    cfg_file = Path("config/config_test_tracking.yaml")

    initialization.print_welcome_msg()

    """ Inizialize Variables """
    # Setup logger
    icepy_utils.setup_logger(log_base_name="rock_tracking")

    # Parse configuration file
    logging.info(f"Configuration file: {cfg_file.stem}")
    cfg = initialization.parse_yaml_cfg(cfg_file)

    timer_global = icepy_utils.AverageTimer()

    init = initialization.Inizialization(cfg)
    init.inizialize_icepy()
    cameras = init.cameras
    cams = init.cams
    features = init.features
    images = init.images
    targets = init.targets
    point_clouds = init.point_clouds
    epoch_dict = init.epoch_dict
    focals = init.focals_dict

    """ Big Loop over epoches """

    logging.info("------------------------------------------------------")
    logging.info("Processing started:")
    timer = icepy_utils.AverageTimer()
    iter = 0
    for epoch in cfg.proc.epoch_to_process:

        logging.info("------------------------------------------------------")
        logging.info(
            f"Processing epoch {epoch} [{iter}/{cfg.proc.epoch_to_process[-1]-cfg.proc.epoch_to_process[0]}] - {epoch_dict[epoch]}..."
        )
        iter += 1

        epochdir = Path(cfg.paths.results_dir) / epoch_dict[epoch]

        # Perform matching and tracking
        if cfg.proc.do_matching:
            features = MatchingAndTracking(
                cfg=cfg,
                epoch=epoch,
                images=images,
                features=features,
                epoch_dict=epoch_dict,
            )
        else:
            try:
                path = epochdir / "matching"
                features[epoch] = load_matches_from_disk()
            except FileNotFoundError as err:
                logging.exception(err)
                logging.warning("Performing new matching and tracking...")
                features = MatchingAndTracking(
                    cfg=cfg,
                    epoch=epoch,
                    images=images,
                    features=features,
                    epoch_dict=epoch_dict,
                )

        timer.update("matching")

        timer.print(f"Epoch {epoch} completed")

    # For debugging

    # Get time series of features
    from typing import TypedDict
    import time

    # def extract_feature_time_series(
    #     fdict: icepy_classes.FeaturesDict,
    #     # pdict: icepy_classes.FeaturesDict,
    #     track_id: np.int32,
    #     min_tracked_epoches: int = 1,
    #     debug: bool = False,
    # ) -> icepy_classes.Feature:
    #     """
    #     extract_feature_time_series _summary_

    #     Args:
    #         fdict (FeaturesDict): _description_
    #         track_id (np.int32): _description_
    #         min_tracked_epoches (int, optional): _description_. Defaults to 1.
    #         debug (bool, optional): _description_. Defaults to False.

    #     Returns:
    #         FeaturesDict: _description_
    #     """

    #     epoches = list(fdict.keys())
    #     if debug:
    #         ts: icepy_classes.Feature = {}
    #         for ep in epoches:
    #             if track_id in fdict[ep]:
    #                 logging.info(f"Feture {track_id} available in ep {ep}")
    #                 ts[ep] = fdict[ep][track_id]
    #             else:
    #                 logging.info(f"Feture {track_id} NOT available in ep {ep}")
    #     else:
    #         ts: icepy_classes.Feature = {
    #             ep: fdict[ep][track_id] for ep in epoches if track_id in fdict[ep]
    #         }

    #     if min_tracked_epoches > 0:
    #         if len(ts) <= min_tracked_epoches:
    #             if debug:
    #                 logging.warning(
    #                     f"Feture {track_id} was detected only in epoch {list(ts.keys())[0]}. Not returned."
    #                 )
    #             return None
    #         else:
    #             return ts
    #     else:
    #         return ts

    # t0 = time.time()
    # out = extract_feature_time_series(fdict, 2, min_tracked_epoches=2)
    # print(f"Elaspsed time {time.time() - t0} s")

    class FeaturePointDict(TypedDict):
        feature: icepy_classes.Feature
        point: icepy_classes.Point

    class TrackedFeaturesDict(TypedDict):
        epoch: FeaturePointDict

    class TrackedFeaturesAll(TypedDict):
        track_id: TrackedFeaturesDict
        # track_id: icepy_classes.FeaturesDict

    cam = cams[0]
    fdict: icepy_classes.Feature = {
        epoch: features[epoch][cam] for epoch in cfg.proc.epoch_to_process
    }

    def extract_feature_time_series(
        fdict: icepy_classes.FeaturesDict,
        # pdict: icepy_classes.FeaturesDict,
        track_id: np.int32,
        min_tracked_epoches: int = 1,
    ) -> TrackedFeaturesDict:
        """
        extract_feature_time_series _summary_

        Args:
            fdict (FeaturesDict): _description_
            track_id (np.int32): _description_
            min_tracked_epoches (int, optional): _description_. Defaults to 1.
            debug (bool, optional): _description_. Defaults to False.

        Returns:
            FeaturesDict: _description_
        """

        epoches = list(fdict.keys())
        ts: TrackedFeaturesDict = {
            ep: fdict[ep][track_id] for ep in epoches if track_id in fdict[ep]
        }

        if min_tracked_epoches > 0:
            if len(ts) <= min_tracked_epoches:
                return None
            else:
                return ts
        else:
            return ts

    # fts: TrackedFeaturesDict = {
    #     0: extract_feature_time_series(fdict, 0, min_tracked_epoches=1),
    #     1: extract_feature_time_series(fdict, 1, min_tracked_epoches=1),
    #     2: extract_feature_time_series(fdict, 2, min_tracked_epoches=1),
    # }

    t0 = time.time()
    last_track_id = fdict[cfg.proc.epoch_to_process[-1]].last_track_id
    fts: TrackedFeaturesDict = {
        id: extract_feature_time_series(fdict, id, min_tracked_epoches=1)
        for id in range(last_track_id)
    }
    print(f"Elaspsed time {time.time() - t0} s")
