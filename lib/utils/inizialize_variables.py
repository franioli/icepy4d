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


from easydict import EasyDict as edict
from pathlib import Path
from typing import List, Union

from lib.base_classes.camera import Camera
from lib.base_classes.pointCloud import PointCloud
from lib.base_classes.images_new import Image, ImageDS
from lib.base_classes.targets import Targets
from lib.base_classes.features import Features


class Inizialization:
    def __init__(self, cfg: edict) -> None:

        self.cfg = cfg
        self.cams = self.cfg.paths.camera_names

    def init_image_ds(self) -> dict:
        # Create Image Datastore objects
        images = dict.fromkeys(self.cams)
        for cam in self.cams:
            images[cam] = ImageDS(self.cfg.paths.image_dir / cam)

        self.images = images
        return self.images

    def init_epoch_dict(self) -> dict:
        epoch_dict = {}
        for epoch in self.cfg.proc.epoch_to_process:
            image = Image(self.images[self.cams[0]].get_image_path(epoch))
            epoch_dict[epoch] = Path(
                (self.cfg.paths.results_dir)
                / f"{image.get_datetime().year}_{image.get_datetime().month:02}_{image.get_datetime().day:02}"
            ).stem

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
    from lib.read_config import parse_yaml_cfg

    cfg_file = "config/config_base.yaml"
    cfg = parse_yaml_cfg(cfg_file)

    init = Inizialization(cfg)
    init.init_image_ds()
    init.init_epoch_dict()
    init.init_cameras()
    init.init_features()
