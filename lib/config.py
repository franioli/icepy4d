import numpy as np
import yaml

from lib.classes import Imageds
from pathlib import Path
from easydict import EasyDict as edict

# This file defines a dictionary, cfg, which includes the default parameters of the pipeline.
# The dictionary is updated/extended at runtime with the parameters defined by the user in the input yaml config file


def parse_yaml_cfg(cfg_file: edict) -> edict:

    with open(cfg_file) as file:
        yaml_opt = edict(yaml.safe_load(file))

    cfg = edict({'paths': edict(), 'proc': edict(),
                'matching': edict(), 'tracking': edict(),
                 'images': edict(), 'georef': edict(),
                 'other': edict(), })

    # - Data paths
    cfg.paths.imdir = Path(yaml_opt.image_dir)
    cfg.paths.caldir = Path(yaml_opt.calibration_dir)
    cfg.paths.resdir = Path(yaml_opt.results_dir)
    cfg.paths.cam_names = yaml_opt.camera_names
    cfg.matching_cfg = Path(yaml_opt.matching_cfg)
    cfg.tracking_cfg = Path(yaml_opt.tracking_cfg)

    # - Processing options
    cfg.proc.epoch_to_process = yaml_opt.epoch_to_process
    cfg.proc.do_matching = yaml_opt.do_matching
    cfg.proc.do_tracking = yaml_opt.do_tracking
    cfg.proc.do_coregistration = yaml_opt.do_coregistration

    # - Matching options
    # @TODO: loop on all elemnts of yaml_opt to create conf dict
    cfg.matching.detector = yaml_opt.detector
    cfg.matching.matcher = yaml_opt.matcher
    cfg.matching.output_dir = yaml_opt.output_dir
    cfg.matching.resize = yaml_opt.resize
    cfg.matching.keypoint_threshold = yaml_opt.keypoint_threshold
    cfg.matching.max_keypoints = yaml_opt.max_keypoints
    cfg.matching.superglue_model = yaml_opt.superglue_model
    cfg.matching.match_threshold = yaml_opt.match_threshold
    cfg.matching.do_viz = yaml_opt.do_viz
    cfg.matching.show_keypoints = yaml_opt.show_keypoints
    cfg.matching.device = yaml_opt.device

    # - Tracking options

    # - Image-realted options
    cfg.images.bbox = np.array(
        yaml_opt.mask_bounding_box).astype('int')

    # - Georef options
    cfg.georef.camera_centers_world = np.array(yaml_opt.camera_centers_world)
    cfg.georef.target_paths = yaml_opt.target_paths

    # - Other options
    cfg.other.do_viz = yaml_opt.do_viz
    cfg.other.do_SOR_filter = yaml_opt.do_SOR_filter

    validate_cfg(cfg)

    print_cfg(cfg)

    return cfg


def validate_cfg(cfg) -> None:
    pass


def print_cfg(cfg) -> None:
    # @TODO: define function for printing main parameters on screen
    pass


def validate_inputs(cfg: edict, images: Imageds) -> edict:

    cams = cfg.paths.cam_names

    # Check that number of images is the same for every camera
    for i in range(1, len(cams)):
        if len(images[cams[i]]) is not len(images[cams[i-1]]):
            raise ValueError('Error: different number of images per camera')
        else:
            print('Image datastores created successfully.')

    # Check and expand epoches to be processed
    if cfg.proc.epoch_to_process == 'all':
        cfg.proc.epoch_to_process = [x for x in range(len(images[cams[0]]))]
    if type(cfg.proc.epoch_to_process) is not list:
        raise ValueError('Invalid input of epoches to process')

    return cfg


if __name__ == '__main__':

    # @TODO: implement parser for setting parameters in command line

    cfg_file = 'config/config_base.yaml'
    parse_yaml_cfg(cfg_file)
