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
import logging
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, TypedDict, Union

import numpy as np
import yaml
from easydict import EasyDict as edict

from icepy4d.classes import (
    Calibration,
    Camera,
    CamerasDict,
    Epoch,
    Features,
    FeaturesDict,
    Image,
    ImageDS,
    ImagesDict,
    PointCloud,
    Points,
    Targets,
)

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


def parse_yaml_cfg(cfg_file: Union[str, Path]) -> edict:
    """
    Parse a YAML configuration file and return it as an easydict.

    Args:
        cfg_file (Union[str, Path]): The path to the configuration file in YAML format.

    Raises:
        ValueError: If the input of epochs to process is invalid.

    Returns:
        edict: A dictionary-like object containing the configuration parameters.
    """

    cfg_file = Path(cfg_file)
    if not cfg_file.exists():
        sys.exit("Configuration file does not exist! Aborting.")

    try:
        with open(cfg_file) as file:
            cfg = edict(yaml.safe_load(file))
    except:
        raise RuntimeError("Unable to create valid cfg dictionary from yaml file")

    # Camera names
    cfg.cams = cfg.paths.camera_names

    # - Data paths
    root_path = Path().absolute()
    cfg.paths.root_path = root_path
    cfg.paths.image_dir = root_path / Path(cfg.paths.image_dir)
    cfg.paths.calibration_dir = root_path / Path(cfg.paths.calibration_dir)
    cfg.paths.results_dir = root_path / Path(cfg.paths.results_dir)

    # - Result paths
    cfg.camea_estimated_fname = cfg.paths.results_dir / "camera_info_est.txt"
    cfg.residuals_fname = cfg.paths.results_dir / "residuals_image.txt"
    cfg.matching_stats_fname = cfg.paths.results_dir / "matching_tracking_results.txt"

    # remove files if they already exist
    if cfg.camea_estimated_fname.exists():
        cfg.camea_estimated_fname.unlink()
    if cfg.residuals_fname.exists():
        cfg.residuals_fname.unlink()
    if cfg.matching_stats_fname.exists():
        cfg.matching_stats_fname.unlink()

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

    # - Check and expand epoches to be processed
    if cfg.proc.epoch_to_process == "all":
        logging.info(
            "Epoch_to_process set to 'all'. Expanding it based on the images found in image folder."
        )
        img_ds = dict.fromkeys(cfg.cams)
        img_ds = ImageDS(cfg.paths.image_dir / cfg.cams[0])
        n_images = len(img_ds)
        cfg.proc.epoch_to_process = [x for x in range(n_images)]
    elif len(cfg.proc.epoch_to_process) == 2:
        logging.info(
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


class Inizializer:
    def __init__(
        self,
        cfg: edict,
    ) -> None:
        """
        __init__ initialization class

        Args:
            cfg (edict): dictionary (as EasyDict object) containing all the configuration parameters.
        """

        print_welcome_msg()

        self.cfg = cfg
        assert (
            "camera_names" in self.cfg.paths
        ), "Camera names not available in cfg file."
        self.cams = self.cfg.paths.camera_names

    def init_image_ds(self):
        """
        init_image_ds _summary_

        Returns:
            ImagesDict: _description_
        """
        self.images = {
            cam: ImageDS(self.cfg.paths.image_dir / cam) for cam in self.cams
        }
        for cam in self.cams:
            self.images[cam].write_exif_to_csv(
                self.cfg.paths.image_dir / f"image_list_{cam}.csv"
            )
        return self.images

    def init_epoch_dict(self):
        """
        init_epoch_dict Build dictonary containing pairs of epoch and dates, as follows:
        {0: "2021_01_01", 1: "2021_01_02" ...}

        Returns:
            EpochDict: epoc_dict
        """
        self.epoch_dict = {}
        for epoch in range(len(self.images[self.cams[0]])):
            date_str = self.images[self.cams[0]].get_image_date(epoch)
            date = datetime.strptime(date_str, "%Y:%m:%d")
            self.epoch_dict[epoch] = f"{date.year}_{date.month:02}_{date.day:02}"
        return self.epoch_dict

    def init_features(self):
        """
        init_features _summary_

        Returns:
            _type_: _description_

        NOTE: This function is deprecated. It is kept for backward compatibility with the function MatchingAndTracking that require the full dictionary of features as input. It will be removed in future versions.
        """
        logging.warning(
            f"This function {__name__} is deprecated. It is kept for backward compatibility with the function MatchingAndTracking that require the full dictionary of features as input. It will be removed in future versions."
        )
        self.features = {}
        for epoch in self.cfg.proc.epoch_to_process:
            self.features[epoch] = {cam: Features() for cam in self.cams}

        return self.features

    def init_epoch(self, epoch_id: int, epoch_dir: Union[Path, str] = None):
        """
        init_epoch _summary_

        Returns:
            Epoch: _description_
        """

        # Build dictionary of Images for the current epoch
        im_epoch: ImagesDict = {
            cam: Image(self.images[cam].get_image_path(epoch_id)) for cam in self.cams
        }

        # Load cameras
        cams_ep: CamerasDict = {}
        for cam in self.cams:
            calib = Calibration(self.cfg.paths.calibration_dir / f"{cam}.txt")
            cams_ep[cam] = calib.to_camera()

        # Load targets
        target_paths = [
            self.cfg.georef.target_dir
            / (im_epoch[cam].stem + self.cfg.georef.target_file_ext)
            for cam in self.cams
        ]
        targ_ep = Targets(
            im_file_path=target_paths,
            obj_file_path=self.cfg.georef.target_dir
            / self.cfg.georef.target_world_file,
        )

        # init empty features and points
        feat_ep = {cam: Features() for cam in self.cams}
        pts_ep = Points()

        epoch = Epoch(
            im_epoch[self.cams[0]].datetime,
            images=im_epoch,
            cameras=cams_ep,
            features=feat_ep,
            points=pts_ep,
            targets=targ_ep,
            point_cloud=None,
            epoch_dir=epoch_dir,
        )
        return epoch


class Inizializer_old:
    def __init__(
        self,
        cfg: edict,
    ) -> None:
        """
        __init__ initialization class

        Args:
            cfg (edict): dictionary (as EasyDict object) containing all the configuration parameters.

        NOTE: This old class do not use Epoch as main container for all the variables. Use Inizializer instead
        """

        print_welcome_msg()

        self.cfg = cfg
        assert (
            "camera_names" in self.cfg.paths
        ), "Camera names not available in cfg file."
        self.cams = self.cfg.paths.camera_names

    def init_image_ds(self):  # -> ImagesDict:
        """
        init_image_ds _summary_

        Returns:
            ImagesDict: _description_
        """
        self.images = {
            cam: ImageDS(self.cfg.paths.image_dir / cam) for cam in self.cams
        }
        for cam in self.cams:
            self.images[cam].write_exif_to_csv(
                self.cfg.paths.image_dir / f"image_list_{cam}.csv"
            )
        return self.images

    def init_epoch_dict(self):
        """
        init_epoch_dict Build dictonary containing pairs of epoch and dates, as follows:
        {0: "2021_01_01", 1: "2021_01_02" ...}

        Returns:
            EpochDict: epoc_dict
        """
        self.epoch_dict = {}
        for epoch in range(len(self.images[self.cams[0]])):
            date_str = self.images[self.cams[0]].get_image_date(epoch)
            date = datetime.strptime(date_str, "%Y:%m:%d")
            self.epoch_dict[epoch] = f"{date.year}_{date.month:02}_{date.day:02}"
        return self.epoch_dict

    def init_cameras(self):
        assert hasattr(
            self, "images"
        ), "Images datastore not available yet. Inizialize images first"
        img = Image(self.images[self.cams[0]].get_image_path(0))
        im_height, im_width = img.height, img.width

        # Inizialize Camera Intrinsics at every epoch setting them equal to the those of the reference cameras.
        self.cameras = {}
        for epoch in self.cfg.proc.epoch_to_process:
            self.cameras[epoch] = {
                cam: Camera(
                    width=im_width,
                    height=im_height,
                    calib_path=self.cfg.paths.calibration_dir / f"{cam}.txt",
                )
                for cam in self.cams
            }

        return self.cameras

    def init_features(self):
        self.features = {}
        for epoch in self.cfg.proc.epoch_to_process:
            self.features[epoch] = {cam: Features() for cam in self.cams}

        return self.features

    def init_targets(self):
        # Read target image coordinates and object coordinates
        self.targets = {}
        for epoch in self.cfg.proc.epoch_to_process:
            p1_path = self.cfg.georef.target_dir / (
                self.images[self.cams[0]].get_image_stem(epoch)
                + self.cfg.georef.target_file_ext
            )

            p2_path = self.cfg.georef.target_dir / (
                self.images[self.cams[1]].get_image_stem(epoch)
                + self.cfg.georef.target_file_ext
            )

            self.targets[epoch] = Targets(
                im_file_path=[p1_path, p2_path],
                obj_file_path=self.cfg.georef.target_dir
                / self.cfg.georef.target_world_file,
            )

        return self.targets

    def init_points(self):
        self.points = {ep: Points() for ep in self.cfg.proc.epoch_to_process}
        return self.points

    def init_point_cloud(self):
        self.point_clouds = {ep: None for ep in self.cfg.proc.epoch_to_process}
        return self.point_clouds

    def init_focals_dict(self) -> dict:
        self.focals_dict = dict.fromkeys(self.cams)
        for cam in self.cams:
            self.focals_dict[cam] = {}


if __name__ == "__main__":
    cfg_file = "./config/config_base.yaml"
    cfg = parse_yaml_cfg(cfg_file)

    print(cfg)

    init = Inizializer(cfg)
    init.init_image_ds()
    init.init_epoch_dict()
    init.init_cameras()
    init.init_features()
