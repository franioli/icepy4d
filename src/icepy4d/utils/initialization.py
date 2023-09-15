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

import argparse
import datetime
import sys
from pathlib import Path
from pprint import pprint
from typing import Tuple, Union, Dict
import os

import numpy as np
import yaml
from easydict import EasyDict as edict

from icepy4d.core import (
    Calibration,
    CamerasDict,
    Epoch,
    EpochDataMap,
    Features,
    ImageDS,
    Points,
    Targets,
)
from icepy4d.utils import setup_logger, get_logger, deprecated

logger = get_logger()

""" 
This file defines the dictionary cfg which includes the default parameters of the pipeline.
The dictionary is updated/extended at runtime with the parameters defined by the user in the input yaml config file
"""


def print_welcome_msg() -> None:
    print("\n================================================================")
    print("ICEpy4D")
    print(
        "Image-based Continuos monitoring of glaciers' Evolution with low-cost stereo-cameras and Deep Learning photogrammetry"
    )
    print("2023 - Francesco Ioli - francesco.ioli@polimi.it")
    print("================================================================\n")


def parse_cfg(cfg_file: Union[str, Path], ignore_errors: bool = False) -> edict:
    """
    Parse a YAML configuration file and return it as an easydict.

    Args:
        cfg_file (Union[str, Path]): The path to the configuration file in YAML format.

    Raises:
        ValueError: If the input of epochs to process is invalid.

    Returns:
        edict: A dictionary-like object containing the configuration parameters.
    """

    print_welcome_msg()

    cfg_file = Path(cfg_file)
    if not cfg_file.exists():
        sys.exit("Configuration file does not exist! Aborting.")

    try:
        with open(cfg_file) as file:
            cfg = edict(yaml.safe_load(file))
    except:
        raise RuntimeError("Unable to create valid cfg dictionary from yaml file")

    # Setup logger
    if "log" not in cfg:
        setup_logger()
    else:
        setup_logger(
            log_folder=cfg.log.get("folder", "logs"),
            log_base_name=cfg.log.get("base_filename", "log"),
            console_log_level=cfg.log.get("log.level", "info"),
            logfile_level=cfg.log.get("log.level", "info"),
        )
    logger = get_logger()
    logger.info(f"Configuration file: {cfg_file.name}")

    # - Data paths
    root_path = Path().absolute()
    cfg.paths.root_path = root_path
    cfg.paths.image_dir = root_path / Path(cfg.paths.image_dir)
    cfg.paths.calibration_dir = root_path / Path(cfg.paths.calibration_dir)
    cfg.paths.results_dir = root_path / Path(cfg.paths.results_dir)

    # Camera names
    cfg.cams = sorted(
        [str(f.name) for f in os.scandir(cfg.paths.image_dir) if f.is_dir()]
    )

    # - Result paths
    cfg.camera_estimated_fname = cfg.paths.results_dir / "camera_info_est.txt"
    cfg.residuals_fname = cfg.paths.results_dir / "residuals_image.txt"
    cfg.matching_stats_fname = cfg.paths.results_dir / "matching_tracking_results.txt"

    # remove files if they already exist
    if not cfg.proc.load_existing_results:
        if cfg.camera_estimated_fname.exists():
            cfg.camera_estimated_fname.unlink()
        if cfg.residuals_fname.exists():
            cfg.residuals_fname.unlink()
        if cfg.matching_stats_fname.exists():
            cfg.matching_stats_fname.unlink()

    # - Image-realted options
    # cfg.images.mask_bounding_box = np.array(cfg.images.mask_bounding_box).astype("int")

    # - Georef options
    cfg.georef.camera_centers_world = np.array(cfg.georef.camera_centers_world)
    cfg.georef.target_dir = Path(cfg.georef.target_dir)

    # - Check and expand epoches to be processed
    if cfg.proc.epoch_to_process == "all":
        logger.info(
            "Epoch_to_process set to 'all'. Expanding it based on the images found in image folder."
        )
        img_ds = dict.fromkeys(cfg.cams)
        img_ds = ImageDS(cfg.paths.image_dir / cfg.cams[0])
        n_images = len(img_ds)
        cfg.proc.epoch_to_process = [x for x in range(n_images)]
    elif len(cfg.proc.epoch_to_process) == 2:
        logger.info(
            f"Epoch_to_process set to a pair of values. Expanding it for a range of epoches from epoch {cfg.proc.epoch_to_process[0]} to {cfg.proc.epoch_to_process[1]}."
        )
        ep_ini = cfg.proc.epoch_to_process[0]
        ep_fin = cfg.proc.epoch_to_process[1]
        cfg.proc.epoch_to_process = [x for x in range(ep_ini, ep_fin)]
    else:
        msg = "Invalid input of epoches to process"
        logger.error(msg)
        raise ValueError(msg)
    assert isinstance(cfg.proc.epoch_to_process, list) and all(
        isinstance(element, int) for element in cfg.proc.epoch_to_process
    ), "Invalid input of epoches to process"

    # if not ignore_errors:
    #     validate_cfg(cfg)

    return cfg


def initialize_epoch(
    cfg: edict, epoch_timestamp: datetime.datetime, images: dict, epoch_dir: Path
):
    """Initialize an Epoch object for ICEPy4D processing.

    This function initializes an Epoch object for processing data in ICEPy4D.
    It sets up the Epoch object with images, cameras, targets, features,
    points, and other necessary data structures for the specified epoch.
    The timestamp of the epoch is taken from the image Exif metadata of the
    first camera in the list of cameras.

    Args:
        cfg (EasyDict): An EasyDict object containing the configuration         parameters, including paths, camera settings, and other relevant data.
        images (Dict[str, Image]): A dictionary containing Image objects, where the keys are camera names (camera keys) and the values are corresponding Image instances.
        epoch_id (int): The epoch ID or index indicating the specific epoch to initialize.
        epoch_dir (Path, optional): The directory path where the epoch data will be stored. If not provided, a directory path will be created in the results directory based on the timestamp of the epoch. Defaults to None.

    Returns:
        Epoch: An initialized Epoch object representing the specified epoch.

    Example:
        >>> cfg = EasyDict()
        >>> # Populate cfg with appropriate configuration parameters
        >>> images = {"cam1": Image(...), "cam2": Image(...)}
        >>> epoch = initialize_epoch(cfg, images, epoch_id=0)
    """

    # Build dictionary of Images for the current epoch
    # im_epoch: ImagesDict = {cam: Image(img) for cam, img in images.items()}

    # Load cameras
    cams_ep: CamerasDict = {}
    for cam in cfg.cams:
        calib = Calibration(cfg.paths.calibration_dir / f"{cam}.txt")
        cams_ep[cam] = calib.to_camera()

    # Load targets
    target_paths = [
        cfg.georef.target_dir / (images[cam].stem + cfg.georef.target_file_ext)
        for cam in cfg.cams
    ]
    targ_ep = Targets(
        im_file_path=target_paths,
        obj_file_path=cfg.georef.target_dir / cfg.georef.target_world_file,
    )

    # init empty features and points
    feat_ep = {cam: Features() for cam in cfg.cams}
    pts_ep = Points()

    epoch_timestamp = epoch_timestamp.replace("_", " ")
    epoch = Epoch(
        epoch_timestamp,
        images=images,
        cameras=cams_ep,
        features=feat_ep,
        points=pts_ep,
        targets=targ_ep,
        point_cloud=None,
        epoch_dir=epoch_dir,
    )
    return epoch


@deprecated
def validate_cfg(cfg: edict) -> None:
    cams = cfg.paths.camera_names
    images = dict.fromkeys(cams)
    for cam in cams:
        images[cam] = ImageDS(cfg.paths.image_dir / cam)

    # Check that number of images is the same for every camera
    for i in range(1, len(cams)):
        if len(images[cams[i]]) != len(images[cams[i - 1]]):
            raise ValueError("Error: different number of images per camera")
        else:
            logger.info("Image datastores created successfully.")


def print_cfg(cfg: edict):
    """
    Print the configuration parameters in a clear and easy-to-read format.

    Args:
        cfg (edict): A dictionary-like object containing the configuration parameters.
    """
    pprint(dict(cfg), indent=2)


def download_model():
    """
    Download the model from the internet.
    """
    pass


@deprecated
def initialize_image_ds(cfg: edict) -> Tuple[Dict[str, ImageDS], EpochDataMap]:
    """Initialize an ImagesDict object with camera directories and metadata.

    This function initializes an ImagesDict object with camera directories and
    metadata for each camera specified in the configuration (cfg). It creates
    an ImageDS object for each camera and writes its corresponding Exif
    metadata to a CSV file.
    Additionally, it creates an EpochDataMap object containing the epoch
    timestamps, taken from the image exif of the first camera in the list of
    cameras.

    Args:
        cfg (EasyDict): An EasyDict object containing the configuration parameters,
            including paths and camera settings.

    Returns:
        Tuple[Dict[str, ImageDS], EpochDataMap]: A tuple containing a dictionary with
            ImageDS objects and an EpochDataMap object. The keys of the dictionary are
            camera names (camera keys), and the values are corresponding ImageDS
            instances. The EpochDataMap object contains the epoch timestamps derived
            from the image Exif metadata of the first camera in the list.

    Example:
        >>> cfg = EasyDict()
        >>> # Populate cfg with appropriate configuration parameters
        >>> images_dict, epoch_dict = initialize_image_ds(cfg)
    """
    images = {cam: ImageDS(cfg.paths.image_dir / cam) for cam in cfg.cams}
    for cam in cfg.cams:
        images[cam].write_exif_to_csv(cfg.paths.image_dir / f"image_list_{cam}.csv")
    epoch_dict = EpochDataMap(images[cfg.cams[0]].timestamps)

    return images, epoch_dict


def parse_command_line() -> Tuple[Path, dict]:
    """
    parse_command_line Parse command line input

    Returns:
        Tuple[str, dict]: Tuple containing the path of the configuration file and a dictionary containing parameters to setup the logger
    """
    parser = argparse.ArgumentParser(
        description="""icepy4d
            Low-cost stereo photogrammetry for 4D glacier monitoring \
            Check -h or --help for options.
        Usage: ./main.py -c config_base.yaml"""
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path of to the configuration file",
    )
    parser.add_argument(
        "--log_folder",
        default="logs",
        type=str,
        help="Folder for storing logs (default: 'logs')",
    )
    parser.add_argument(
        "--log_name",
        default="log",
        type=str,
        help="Base name of the log file",
    )
    parser.add_argument(
        "--log_file_level",
        default="info",
        type=str,
        help="Set log level for logging to file \
            (possible options are: 'debug', 'info', \
            'warning', 'error', 'critical')",
    )
    parser.add_argument(
        "--log_console_level",
        default="info",
        type=str,
        help="Set log level for logging to stdout \
            (possible options are: 'debug', 'info', \
            'warning', 'error', 'critical')",
    )
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        raise ValueError(
            "Not enough input arguments. Specify at least the configuration file. Use --help (or -h) for help."
        )

    cfg_file = Path(args.config)
    log_cfg = {
        "log_folder": args.log_folder,
        "log_name": args.log_name,
        "log_file_level": args.log_file_level,
        "log_console_level": args.log_console_level,
    }

    return cfg_file, log_cfg


if __name__ == "__main__":
    from icepy4d.core.epoch import Epoch, Epoches

    cfg_file = "./config/config_base.yaml"
    cfg = parse_cfg(cfg_file)
    print(cfg)
    images, epoch_dict = initialize_image_ds(cfg)
    a = images["p1"]
    timestamps = a.timestamps

    epoch = initialize_epoch(cfg, images, epoch_id=0)
    epoches = Epoches()
    epoches.add_epoch(epoch)

    # init = initializer(cfg)
    # init.init_image_ds()
    # init.init_epoch_dict()
    # init.init_cameras()
    # init.init_features()

    print("Done")
