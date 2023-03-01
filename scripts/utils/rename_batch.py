""" Batch rename multiple files """

from pathlib import Path
from typing import Union


def str_replace(
    dir: Union[str, Path], src: str, dst: str, file_pattern: str = None
) -> None:

    if file_pattern is None:
        file_pattern = "*"

    dir = Path(dir)
    for file in sorted(dir.glob(file_pattern)):
        name = file.name
        file.rename(file.parent / name.replace(src, dst))


def str_replace_posix(
    dir: Union[str, Path], src: str, dst: str, file_pattern: str = None
) -> None:

    if file_pattern is None:
        file_pattern = "*"

    dir = Path(dir)
    for file in sorted(dir.glob(file_pattern)):
        name = file.name
        file.rename(file.parent / name.replace(src, dst))


if __name__ == "__main__":

    folder = "res/point_clouds"
    file_pattern = f"sparse_*"

    src_pattern = ""
    dst_pattern = ""

    str_replace(dir=folder, src=src_pattern, dst=dst_pattern, file_pattern=file_pattern)
