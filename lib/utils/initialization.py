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

from easydict import EasyDict as edict
from pathlib import Path
from typing import List, Union
from datetime import datetime

from lib.base_classes.camera import Camera
from lib.base_classes.pointCloud import PointCloud
from lib.base_classes.images import Image, ImageDS
from lib.base_classes.targets import Targets
from lib.base_classes.features import Features


# This file defines the dictionary cfg which includes the default parameters of the pipeline.
# The dictionary is updated/extended at runtime with the parameters defined by the user in the input yaml config file


def parse_yaml_cfg(cfg_file: Union[str, Path]) -> edict:

    with open(cfg_file) as file:
        cfg = edict(yaml.safe_load(file))

    # - Data paths
    root_path = Path().absolute()
    cfg.paths.root_path = root_path
    cfg.paths.image_dir = root_path / Path(cfg.paths.image_dir)
    cfg.paths.calibration_dir = root_path / Path(cfg.paths.calibration_dir)
    cfg.paths.results_dir = root_path / Path(cfg.paths.results_dir)
    # cfg.paths.last_match_path = root_path / Path(cfg.paths.last_match_path)--> Deprecated

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
            "Epoch_to_process set to a pair of values. Expanding it for a range of epoches from the first to the second."
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
            print("Image datastores created successfully.")


def print_cfg(cfg) -> None:
    # TODO: implement printing of configuration
    for key, value in cfg.items():
        print(key + " : " + str(value))


def build_metashape_cfg(cfg: edict, epoch_dict: dict, epoch: int) -> edict:

    ms_cfg = edict()
    
    return ms_cfg


class Inizialization:
    def __init__(
        self,
        cfg: edict,
    ) -> None:

        self.cfg = cfg
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
        self.focals_dict = {0: [], 1: []}

    def inizialize_belpy(self) -> dict:
        self.init_image_ds()
        self.init_epoch_dict()
        self.init_cameras()
        self.init_features()
        self.init_targets()
        self.init_point_cloud()
        self.init_focals_dict()


if __name__ == "__main__":

    # @TODO: implement parser for setting parameters in command line

    cfg_file = "./config/config_base.yaml"
    cfg = parse_yaml_cfg(cfg_file)

    print(cfg)

    init = Inizialization(cfg)
    init.init_image_ds()
    init.init_epoch_dict()
    init.init_cameras()
    init.init_features()
