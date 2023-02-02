import os
import pytest

from pathlib import Path
import tempfile

# The path to our assets directory
@pytest.fixture
def data_dir():
    return Path(os.path.split(__file__)[0]).parent / "assets"


@pytest.fixture
def log_dir():
    dirpath = tempfile.mkdtemp()
    return dirpath


@pytest.fixture
def cfg_file(data_dir):
    return data_dir / "config.yaml"


# @pytest.fixture
# def epoch_dict(data_dir):
