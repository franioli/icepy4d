"""
MIT License

Copyright (c) 2022 Francesco Ioli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import yaml
import logging

from pathlib import Path
from easydict import EasyDict as edict
from typing import List

from lib.base_classes.images import ImageDS

# This file defines the dictionary cfg which includes the default parameters of the pipeline.
# The dictionary is updated/extended at runtime with the parameters defined by the user in the input yaml config file


def parse_yaml_cfg(cfg_file: edict, logger: logging = None) -> edict:

    with open(cfg_file) as file:
        cfg = edict(yaml.safe_load(file))

    # - Data paths
    root_path = Path().absolute()
    cfg.paths.root_path = root_path
    cfg.paths.image_dir = root_path / Path(cfg.paths.image_dir)
    cfg.paths.calibration_dir = root_path / Path(cfg.paths.calibration_dir)
    cfg.paths.results_dir = root_path / Path(cfg.paths.results_dir)
    cfg.paths.last_match_path = root_path / Path(cfg.paths.last_match_path)

    # - Processing options
    if cfg.proc.do_matching == False and cfg.proc.do_tracking == True:
        msg = "Warning: Invalid combination of options between Matching and Tracking. Tracking was se to enabled, but Matching was not. Disabling Tracking."
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)

        cfg.proc.do_tracking == False

    # - Image-realted options
    cfg.images.mask_bounding_box = np.array(cfg.images.mask_bounding_box).astype("int")

    # - Georef options
    cfg.georef.camera_centers_world = np.array(cfg.georef.camera_centers_world)
    cfg.georef.target_dir = Path(cfg.georef.target_dir)

    # Check and expand epoches to be processed
    if cfg.proc.epoch_to_process == "all":
        if logger is not None:
            logger.warning(
                "Epoch_to_process set to 'all'. Expanding it based on the images found in image folder."
            )
        cams = cfg.paths.camera_names
        img_ds = dict.fromkeys(cams)
        img_ds = ImageDS(cfg.paths.image_dir / cams[0])
        n_images = len(img_ds)
        cfg.proc.epoch_to_process = [x for x in range(n_images)]
    elif len(cfg.proc.epoch_to_process) == 2:
        if logger is not None:
            logger.warning(
                "Epoch_to_process set to a pair of values. Expanding it for a range of epoches from the first to the second."
            )
        ep_ini = cfg.proc.epoch_to_process[0]
        ep_fin = cfg.proc.epoch_to_process[1]
        cfg.proc.epoch_to_process = [x for x in range(ep_ini, ep_fin)]
    else:
        raise ValueError("Invalid input of epoches to process")
    assert isinstance(cfg.proc.epoch_to_process, list) and all(
        isinstance(element, int) for element in cfg.proc.epoch_to_process
    ), "Invalid input of epoches to process"

    validate_cfg(cfg)

    return cfg


def validate_cfg(cfg: edict) -> None:
    cams = cfg.paths.camera_names
    images = dict.fromkeys(cams)
    for cam in cams:
        images[cam] = ImageDS(cfg.paths.image_dir / cam)

    # Check that number of images is the same for every camera
    for i in range(1, len(cams)):
        if len(images[cams[i]]) is not len(images[cams[i - 1]]):
            raise ValueError("Error: different number of images per camera")
        else:
            print("Image datastores created successfully.")


def print_cfg(cfg) -> None:
    for key, value in cfg.items():
        print(key + " : " + str(value))


if __name__ == "__main__":

    # @TODO: implement parser for setting parameters in command line

    cfg_file = "./config/config_base.yaml"
    cfg = parse_yaml_cfg(cfg_file)

    print(cfg)

    cfg.matching.output_dir
