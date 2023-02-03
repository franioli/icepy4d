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
import sys
import argparse

from easydict import EasyDict as edict
from pathlib import Path
from typing import List, Union, Tuple
from datetime import datetime

from ..classes.camera import Camera
from ..classes.features import Features
from ..classes.point_cloud import PointCloud
from ..classes.images import Image, ImageDS
from ..classes.targets import Targets


""" 
This file defines the dictionary cfg which includes the default parameters of the pipeline.
The dictionary is updated/extended at runtime with the parameters defined by the user in the input yaml config file
"""


def print_welcome_msg() -> None:
    print("\n===========================================================")
    print("ICEpy4D")
    print(
        "Image-based Continuos monitoring of glaciers' Evolution with low-cost stereo-cameras and Deep Learning photogrammetry"
    )
    print("2022 - Francesco Ioli - francesco.ioli@polimi.it")
    print("===========================================================\n")


def parse_command_line() -> Tuple[str, dict]:
    """
    parse_command_line Parse command line input

    Returns:
        Tuple[str, dict]: Tuple containing the path of the configuration file and a dictionary containing parameters to setup the logger
    """
    parser = argparse.ArgumentParser(
        description="""icepy
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
        default="icepy",
        type=str,
        help="",
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
    if not cfg_file.exists():
        sys.exit("Configuration file does not exist! Aborting...")

    log_cfg = {
        "log_folder": args.log_folder,
        "log_name": args.log_folder,
        "log_file_level": args.log_file_level,
        "log_console_level": args.log_console_level,
    }

    return cfg_file, log_cfg


def parse_yaml_cfg(cfg_file: Union[str, Path]) -> edict:
    """
    parse_yaml_cfg _summary_

    Args:
        cfg_file (Union[str, Path]): _description_

    Raises:
        ValueError: _description_

    Returns:
        edict: _description_
    """

    with open(cfg_file) as file:
        cfg = edict(yaml.safe_load(file))
    assert isinstance(
        cfg, edict
    ), "Unable to create valid cfg dictionary from yaml file"

    # - Data paths
    root_path = Path().absolute()
    cfg.paths.root_path = root_path
    cfg.paths.image_dir = root_path / Path(cfg.paths.image_dir)
    cfg.paths.calibration_dir = root_path / Path(cfg.paths.calibration_dir)
    cfg.paths.results_dir = root_path / Path(cfg.paths.results_dir)

    # - Processing options
    if cfg.proc.do_matching == False and cfg.proc.do_tracking == True:
        logging.warning(
            "Invalid combination of Matching and Tracking options. Tracking was se to enabled, but Matching was not. Disabling Tracking."
        )
        cfg.proc.do_tracking == False

    # - Image-realted options
    cfg.images.mask_bounding_box = np.array(cfg.images.mask_bounding_box).astype("int")

    # - Georef options
    cfg.georef.camera_centers_world = np.array(cfg.georef.camera_centers_world)
    cfg.georef.target_dir = Path(cfg.georef.target_dir)

    # Check and expand epoches to be processed
    if cfg.proc.epoch_to_process == "all":
        logging.warning(
            "Epoch_to_process set to 'all'. Expanding it based on the images found in image folder."
        )
        cams = cfg.paths.camera_names
        img_ds = dict.fromkeys(cams)
        img_ds = ImageDS(cfg.paths.image_dir / cams[0])
        n_images = len(img_ds)
        cfg.proc.epoch_to_process = [x for x in range(n_images)]
    elif len(cfg.proc.epoch_to_process) == 2:
        logging.warning(
            f"Epoch_to_process set to a pair of values. Expanding it for a range of epoches from epoch {cfg.proc.epoch_to_process[0]} to {cfg.proc.epoch_to_process[1]}."
        )
        ep_ini = cfg.proc.epoch_to_process[0]
        ep_fin = cfg.proc.epoch_to_process[1]
        cfg.proc.epoch_to_process = [x for x in range(ep_ini, ep_fin)]
    else:
        msg = "Invalid input of epoches to process"
        logging.error(msg)
        raise ValueError(msg)
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
        if len(images[cams[i]]) != len(images[cams[i - 1]]):
            raise ValueError("Error: different number of images per camera")
        else:
            logging.info("Image datastores created successfully.")


def print_cfg(cfg) -> None:
    # TODO: implement printing of configuration
    for key, value in cfg.items():
        print(key + " : " + str(value))


class Inizialization:
    def __init__(
        self,
        cfg: edict,
    ) -> None:
        """
        __init__ initialization class

        Args:
            cfg (edict): dictionary (as EasyDict object) containing all the configuration parameters.
        """

        self.cfg = cfg
        assert (
            "camera_names" in self.cfg.paths
        ), "Camera names not available in cfg file."
        self.cams = self.cfg.paths.camera_names

    def init_image_ds(self) -> dict:
        # Create Image Datastore objects
        images = dict.fromkeys(self.cams)
        for cam in self.cams:
            images[cam] = ImageDS(self.cfg.paths.image_dir / cam)
            images[cam].write_exif_to_csv(
                self.cfg.paths.image_dir / f"image_list_{cam}.csv"
            )

        self.images = images
        return self.images

    def init_epoch_dict_old(self) -> dict:
        epoch_dict = {}
        for epoch in self.cfg.proc.epoch_to_process:
            image = Image(self.images[self.cams[0]].get_image_path(epoch))
            epoch_dict[epoch] = Path(
                (self.cfg.paths.results_dir)
                / f"{image.get_datetime().year}_{image.get_datetime().month:02}_{image.get_datetime().day:02}"
            ).stem

        self.epoch_dict = epoch_dict
        return self.epoch_dict

    def init_epoch_dict(self) -> dict:
        """
        init_epoch_dict Build dictonary containing pairs of epoch and dates, as follows:
        {0: "2021_01_01", 1: "2021_01_02" ...}

        Returns:
            dict: epoc_dict
        """
        epoch_dict = {}
        for epoch in range(len(self.images[self.cams[0]])):
            date_str = self.images[self.cams[0]].get_image_date(epoch)
            date = datetime.strptime(date_str, "%Y:%m:%d")
            epoch_dict[epoch] = f"{date.year}_{date.month:02}_{date.day:02}"
        self.epoch_dict = epoch_dict
        return self.epoch_dict

    def init_cameras(self) -> dict:
        cameras = {}

        assert hasattr(
            self, "images"
        ), "Images datastore not available yet. Inizialize images first"
        img = Image(self.images[self.cams[0]].get_image_path(0))
        im_height, im_width = img.height, img.width

        # Inizialize Camera Intrinsics at every epoch setting them equal to the those of the reference cameras.
        for epoch in self.cfg.proc.epoch_to_process:
            cameras[epoch] = {}
            for cam in self.cams:
                cameras[epoch][cam] = Camera(
                    width=im_width,
                    height=im_height,
                    calib_path=self.cfg.paths.calibration_dir / f"{cam}.txt",
                )

        self.cameras = cameras
        return self.cameras

    def init_features(self) -> dict:
        features = {}

        for epoch in self.cfg.proc.epoch_to_process:
            features[epoch] = dict.fromkeys(self.cams)
            for cam in self.cams:
                features[epoch][cam] = Features()

        self.features = features
        return self.features

    def init_targets(self) -> dict:
        # Read target image coordinates and object coordinates
        targets = {}
        for epoch in self.cfg.proc.epoch_to_process:

            p1_path = self.cfg.georef.target_dir / (
                self.images[self.cams[0]].get_image_stem(epoch)
                + self.cfg.georef.target_file_ext
            )

            p2_path = self.cfg.georef.target_dir / (
                self.images[self.cams[1]].get_image_stem(epoch)
                + self.cfg.georef.target_file_ext
            )

            targets[epoch] = Targets(
                im_file_path=[p1_path, p2_path],
                obj_file_path=self.cfg.georef.target_dir
                / self.cfg.georef.target_world_file,
            )

        self.targets = targets
        return self.targets

    def init_point_cloud(self) -> List[PointCloud]:
        self.point_clouds = {}
        return self.point_clouds

    def init_focals_dict(self) -> dict:
        self.focals_dict = dict.fromkeys(self.cams)
        for cam in self.cams:
            self.focals_dict[cam] = {}

    def inizialize_icepy(self) -> dict:
        self.init_image_ds()
        self.init_epoch_dict()
        self.init_cameras()
        self.init_features()
        self.init_targets()
        self.init_point_cloud()
        self.init_focals_dict()


if __name__ == "__main__":

    cfg_file = "./config/config_base.yaml"
    cfg = parse_yaml_cfg(cfg_file)

    print(cfg)

    init = Inizialization(cfg)
    init.init_image_ds()
    init.init_epoch_dict()
    init.init_cameras()
    init.init_features()
