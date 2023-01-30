import pytest
import os

from pathlib import Path
from easydict import EasyDict as edict

from src.icepy.utils import initialization


def test_inizalization(cfg_file):
    cfg = initialization.parse_yaml_cfg(cfg_file)
    assert isinstance(
        cfg, edict
    ), "Unable to create valid cfg dictionary from yaml file"
    assert (
        cfg.paths.image_dir == Path(os.path.split(__file__)[0]).parent / "assets/img"
    ), "Invalid image path from yaml file"
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
