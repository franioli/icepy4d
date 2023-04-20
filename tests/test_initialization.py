import pytest
import sys

from datetime import datetime
from pathlib import Path
from easydict import EasyDict as edict

from icepy4d.classes import ImageDS
from icepy4d.utils.initialization import (
    parse_command_line,
    parse_yaml_cfg,
    Inizializer,
)


def test_parse_command_line():
    # Test default config
    sys.argv = ["./main.py", "-c", "config.yaml"]
    cfg_file, log_cfg = parse_command_line()
    assert cfg_file == Path("config.yaml")
    assert log_cfg == {
        "log_folder": "logs",
        "log_name": "log",
        "log_file_level": "info",
        "log_console_level": "info",
    }

    # Test custom config
    sys.argv = [
        "./main.py",
        "-c",
        "config.yaml",
        "--log_folder",
        "log",
        "--log_name",
        "new_log",
        "--log_file_level",
        "debug",
        "--log_console_level",
        "error",
    ]
    cfg_file, log_cfg = parse_command_line()
    assert cfg_file == Path("config.yaml")
    assert log_cfg == {
        "log_folder": "log",
        "log_name": "new_log",
        "log_file_level": "debug",
        "log_console_level": "error",
    }

    with pytest.raises(ValueError, match="Not enough input arguments"):
        sys.argv = ["./main.py"]
        parse_command_line()


def test_parse_yaml_cfg(data_dir, cfg_file):
    with pytest.raises(
        SystemExit, match="Configuration file does not exist! Aborting."
    ):
        parse_yaml_cfg("non_existent_config.yaml")

    cfg = parse_yaml_cfg(cfg_file)
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


def test_inizialization_epoch_dict(cfg_file):
    cfg = parse_yaml_cfg(cfg_file)
    init = Inizializer(cfg)
    init.init_image_ds()
    epoch_dict = init.init_epoch_dict()
    assert isinstance(epoch_dict, dict), "Unable to build epoch_dict dictionary"
    true_dict = {0: "2022_05_01", 1: "2022_05_11", 2: "2022_05_18", 3: "2022_05_26"}
    assert epoch_dict == true_dict, "Unable to build epoch_dict dictionary"
