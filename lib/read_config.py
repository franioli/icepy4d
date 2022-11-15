import numpy as np
import yaml

from pathlib import Path
from easydict import EasyDict as edict

from lib.classes import Imageds

# from classes import Imageds

# This file defines a dictionary, cfg, which includes the default parameters of the pipeline.
# The dictionary is updated/extended at runtime with the parameters defined by the user in the input yaml config file


def parse_yaml_cfg(cfg_file: edict) -> edict:

    with open(cfg_file) as file:
        cfg = edict(yaml.safe_load(file))

    # - Data paths
    cfg.paths.image_dir = Path(cfg.paths.image_dir)
    cfg.paths.calibration_dir = Path(cfg.paths.calibration_dir)
    cfg.paths.results_dir = Path(cfg.paths.results_dir)

    # - Image-realted options
    cfg.images.mask_bounding_box = np.array(cfg.images.mask_bounding_box).astype("int")

    # - Georef options
    cfg.georef.camera_centers_world = np.array(cfg.georef.camera_centers_world)
    cfg.georef.target_dir = Path(cfg.georef.target_dir)

    # Check and expand epoches to be processed
    if cfg.proc.epoch_to_process == "all":
        cams = cfg.paths.camera_names
        img_ds = dict.fromkeys(cams)
        img_ds = Imageds(cfg.paths.image_dir / cams[0])
        n_images = len(img_ds)
        cfg.proc.epoch_to_process = [x for x in range(n_images)]
    if type(cfg.proc.epoch_to_process) is not list:
        raise ValueError("Invalid input of epoches to process")

    validate_cfg(cfg)

    return cfg


def validate_cfg(cfg: edict) -> None:
    cams = cfg.paths.camera_names
    images = dict.fromkeys(cams)
    for cam in cams:
        images[cam] = Imageds(cfg.paths.image_dir / cam)

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
