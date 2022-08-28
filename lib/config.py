import numpy as np
import yaml

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

    # - Processing options
    cfg.proc.epoch_to_process = yaml_opt.epoch_to_process
    cfg.proc.do_matching = yaml_opt.do_matching
    cfg.proc.do_tracking = yaml_opt.do_tracking
    cfg.proc.do_coregistration = yaml_opt.do_coregistration


    # - Matching options
    # @TODO: simplify matching options
    cfg.matching.keypoints = yaml_opt.keypoints
    cfg.matching.matcher = yaml_opt.matcher
    cfg.matching.output_dir: yaml_opt.output_dir
    cfg.matching.resize: yaml_opt.resize
    cfg.matching.resize_float: yaml_opt.resize_float
    cfg.matching.equalize_hist: yaml_opt.equalize_hist
    cfg.matching.nms_radius: yaml_opt.nms_radius
    cfg.matching.keypoint_threshold: yaml_opt.keypoint_threshold
    cfg.matching.max_keypoints: yaml_opt.max_keypoints
    cfg.matching.superglue: yaml_opt.superglue
    cfg.matching.sinkhorn_iterations: yaml_opt.sinkhorn_iterations
    cfg.matching.match_threshold: yaml_opt.match_threshold
    cfg.matching.viz: yaml_opt.viz
    cfg.matching.viz_extension: yaml_opt.viz_extension
    cfg.matching.fast_viz: yaml_opt.fast_viz
    cfg.matching.opencv_display: yaml_opt.opencv_display
    cfg.matching.show_keypoints: yaml_opt.show_keypoints
    cfg.matching.cache: yaml_opt.cache
    cfg.matching.force_cpu: yaml_opt.force_cpu
    cfg.matching.useTile: yaml_opt.useTile
    cfg.matching.writeTile2Disk: yaml_opt.writeTile2Disk
    cfg.matching.do_viz_tile: yaml_opt.do_viz_tile
    cfg.matching.rowDivisor: yaml_opt.rowDivisor
    cfg.matching.colDivisor: yaml_opt.colDivisor
    cfg.matching.overlap: yaml_opt.overlap
    
    # - Tracking options

    # - Image-realted options
    cfg.images.mask_bounding_box = np.array(
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
    print('Input parameters are valid.')


def print_cfg(cfg) -> None:
    # @TODO: define function for printing main parameters on screen
    print('')


if __name__ == '__main__':

    # @TODO: implement parser for setting parameters in command line

    cfg_file = 'config/config_base.yaml'
    parse_yaml_cfg(cfg_file)
