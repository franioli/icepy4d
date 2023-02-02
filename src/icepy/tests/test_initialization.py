import pytest
import os

from pathlib import Path
from easydict import EasyDict as edict

from ..utils import initialization


def test_parse_yaml_cfg(data_dir, cfg_file):
    cfg = initialization.parse_yaml_cfg(cfg_file)
    assert isinstance(
        cfg, edict
    ), "Unable to create valid cfg dictionary from yaml file"
    assert cfg.paths.image_dir == data_dir / "img", "Invalid image path from yaml file"
    assert cfg.paths.camera_names == [
        "cam1",
        "cam2",
    ], "Unable to read camera_names list"
    assert cfg.proc.epoch_to_process == [
        0,
        1,
        2,
        3,
    ], "Unable to expand epoch_to_process from pair of values"


def test_inizialization(cfg_file):
    cfg = initialization.parse_yaml_cfg(cfg_file)
    init = initialization.Inizialization(cfg)
    init.init_image_ds()
    epoch_dict = init.init_epoch_dict()
    assert isinstance(epoch_dict, dict), "Unable to build epoch_dict dictionary"
    true_dict = {0: "2022_05_01", 1: "2022_05_11", 2: "2022_05_18", 3: "2022_05_26"}
    assert epoch_dict == true_dict, "Unable to build epoch_dict dictionary"
