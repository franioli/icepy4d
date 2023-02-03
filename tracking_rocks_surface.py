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

    # Setup logger
    icepy_utils.setup_logger("icepy")

    print("\n===========================================================")
    print("ICEpy4D")
    print(
        "Image-based Continuos monitoring of glaciers' Evolution with low-cost stereo-cameras and Deep Learning photogrammetry"
    )
    print("2022 - Francesco Ioli - francesco.ioli@polimi.it")
    print("===========================================================\n")

    # Read options from yaml file
    logging.info(f"Configuration file: {cfg_file.stem}")
    cfg = initialization.parse_yaml_cfg(cfg_file)

    """ Inizialize Variables """

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

        # Testing
        # a = features[180]["p1"]
        # b = features[181]["p1"]
