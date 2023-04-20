import numpy as np
import argparse

from pathlib import Path


def retrieve_matches(opt: dict):
    """
    retrieve_matches Retrieve matches saved by SuperGlue in a npz archive and save results in text files

    Args:
        opt (dict): dict with the following keys:
            - npz (str): path to the npz file
            - output_dir (str): output direcotry path
    Raises:
        IOError: unable to read npz file
    """

    npz_path = Path(opt.npz)
    output_path = Path(opt.output_dir)

    npz_path = Path(npz_path)
    if npz_path.exists():
        try:
            npz = np.load(npz_path)
        except:
            raise IOError(f"Cannot load matches .npz file {npz_path.name}")
    if "npz" in locals():
        print(f"Data in {npz_path.name} loaded successfully")
    else:
        print("No data loaded")

    output_path = Path(output_path)
    if not output_path.is_dir():
        output_path.mkdir()

    res = {k: npz[k] for k in npz.keys()}
    for k in npz.keys():
        np.savetxt(
            output_path.as_posix() + "/" + k + ".txt",
            res[k],
            fmt="%.2f",
            delimiter=",",
            newline="\n",
        )
    print("Data saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve SuperGlue matches from npz archive"
    )
    parser.add_argument("--npz", metavar="<STR>", type=str, help="input npz file")
    parser.add_argument(
        "--output_dir", metavar="<STR>", type=str, help="Output directory"
    )
    opt = parser.parse_opt()

    retrieve_matches(opt)
