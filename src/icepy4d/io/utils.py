import logging
import os
from pathlib import Path
from shutil import copy as scopy
from typing import Union

logger = logging.getLogger(__name__)

def make_symlink(src: Union[str, Path], dst: Union[str, Path], force_overwrite=False):
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source file {src} does not exist.")
    if not dst.exists():
        try:
            os.symlink(src, dst)
        except OSError:
            logger.warning(f"Unable to create symbolic link to {dst}. Copying file.")
            scopy(src, dst)
    elif force_overwrite:
        os.remove(dst)
        try:
            os.symlink(src, dst)
            logger.warning(f"Symbolic link overwritten.")
        except OSError:
            logger.warning("Unable to create symbolic link. Copying file instead.")
            scopy(src, dst)
    else:
        logger.warning(f"{dst} already exists. Skipping symbolic link creation.")


def create_directory(path):
    """
    Creates a directory, if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
