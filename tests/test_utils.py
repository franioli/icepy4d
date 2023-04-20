import pytest
import os

from pathlib import Path

from icepy4d.utils import setup_logger


def test_logger(log_dir):
    log_name = "icepy4d"
    log_file_level = "info"
    log_console_level = "info"
    try:
        setup_logger(log_dir, log_name, log_file_level, log_console_level)
    except RuntimeError as err:
        assert False, f"Unable to set up logger"
    with pytest.raises(RuntimeError) as exc:
        log_file_level = "ino"
        setup_logger(log_dir, log_name, log_file_level, log_console_level)
