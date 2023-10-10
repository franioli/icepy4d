from pathlib import Path
from impreproc import Image


def renaming_file(file: Path, new_name: str) -> bool:
    """Rename a file"""
    file = Path(file)
    assert file.exists(), f"File {file} does not exist"
    assert file.is_file(), f"{file} is not a file"
    assert new_name != "", "New name cannot be empty"
    new_file = file.parent / new_name
    file.rename(new_file)
    return bool


def new_name_from_timestamp(path: Path, prefix: str) -> str:
    """Create a new name from timestamp"""
    img = Image(path)
    timestamp = img._date_time.strftime("%Y%m%d_%H%M%S")
    new_name = f"{prefix}_{timestamp}_{img.path.stem}{img.path.suffix.lower()}"
    return new_name


def new_name_from_corresponding_img(name: Path, imlist):
    for img in imlist:
        if name.stem in img.name:
            return f"{img.stem}{name.suffix}"
    return None


if __name__ == "__main__":
    root_dir = Path(
        "/home/francesco/Insync/10454429@polimi.it/OneDrive Biz - Shared/summer-school-belvedere/stereo_processing/"
    )

    # folder = root_dir / "data/img/p2"
    # prefix = "p2"
    # for file in sorted(Path(folder).glob("*")):
    #     new_name = new_name_from_timestamp(file, prefix)
    #     renaming_file(file, new_name)

    # folder = root_dir / "data/img/p1"
    # prefix = "p1"
    # for file in sorted(Path(folder).glob("*")):
    #     new_name = new_name_from_timestamp(file, prefix)
    #     renaming_file(file, new_name)

    target_folder = root_dir / "data/targets"
    img_folder = root_dir / "data/img/p2"
    imlist = sorted(Path(img_folder).glob("*"))
    for target in sorted(Path(target_folder).glob("*")):
        new_name = new_name_from_corresponding_img(target, imlist)
        if new_name is not None:
            renaming_file(target, new_name)

    print("done")
