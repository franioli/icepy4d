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
    cfg.images.mask_bounding_box = np.array(cfg.images.mask_bounding_box).astype('int')

    # - Georef options
    cfg.georef.camera_centers_world = np.array(cfg.georef.camera_centers_world)
    cfg.georef.target_dir = Path(cfg.georef.target_dir)

    # Check and expand epoches to be processed
    if cfg.proc.epoch_to_process == 'all':
        cams = cfg.paths.camera_names
        images = dict.fromkeys(cams)
        cfg.proc.epoch_to_process = [x for x in range(len(images[cams[0]]))]
    if type(cfg.proc.epoch_to_process) is not list:
        raise ValueError('Invalid input of epoches to process')

    validate_cfg(cfg)

    # cfg = edict({'paths': edict(), 'proc': edict(),
    #             'matching': edict(), 'tracking': edict(),
    #              'images': edict(), 'georef': edict(),
    #              'other': edict(), })

    # # - Processing options
    # cfg.proc.epoch_to_process = yaml_opt.epoch_to_process
    # cfg.proc.do_matching = yaml_opt.do_matching
    # cfg.proc.do_tracking = yaml_opt.do_tracking
    # cfg.proc.do_coregistration = yaml_opt.do_coregistration

    # # - Image-realted options
    # cfg.images.bbox = np.array(
    #     yaml_opt.mask_bounding_box).astype('int')

    # # - Georef options
    # cfg.georef.camera_centers_world = np.array(yaml_opt.camera_centers_world)
    # cfg.georef.target_dir = Path(yaml_opt.target_dir)
    # cfg.georef.target_file_ext = yaml_opt.target_file_ext

    # # - Other options
    # cfg.other.do_viz = yaml_opt.do_viz
    # cfg.other.do_SOR_filter = yaml_opt.do_SOR_filter

    # validate_cfg(cfg)
    # print_cfg(cfg)

    return cfg


def validate_cfg(cfg: edict) -> None:
    cams = cfg.paths.camera_names
    images = dict.fromkeys(cams)
    for cam in cams:
        images[cam] = Imageds(cfg.paths.image_dir / cam)

    # Check that number of images is the same for every camera
    for i in range(1, len(cams)):
        if len(images[cams[i]]) is not len(images[cams[i-1]]):
            raise ValueError('Error: different number of images per camera')
        else:
            print('Image datastores created successfully.')


def print_cfg(cfg) -> None:
    for key, value in cfg.items():
        print(key + " : " + str(value))


if __name__ == '__main__':

    # @TODO: implement parser for setting parameters in command line

    cfg_file = "./config/config_base.yaml"
    cfg = parse_yaml_cfg(cfg_file)

    print(cfg)

    cfg.matching.output_dir
