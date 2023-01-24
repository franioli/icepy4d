"""
Require CloudComPy310 environment set up and activated

"""

import gc
import time
import logging
import json
import numpy as np

from easydict import EasyDict as edict
from multiprocessing import Pool, current_process
from itertools import repeat
from pathlib import Path
from typing import Tuple

from src.belpy.pcd_proc.cloudcompare import DemOfDifference, make_pairs


PCD_DIR = "res/point_clouds_meshed"
PCD_PATTERN = "sampled*.ply"
OUT_DIR = "res/volumes_variations"
DOD_DIR = "z"
TSTEP = 5
GRID_STEP = 0.2


LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)


def DOD_task(
    pcd_pair: Tuple,
    cfg: dict,
) -> None:
    """
    DOD_task run DOD task for parallel computing

    Args:
        pcd_pair (Tuple): Tuple containing the path of the two point clouds.
        cfg (dict): configuration dictionary with at least the following keys:
                {
                    "grid_step" (float): cell size of the rasterized DEM (default = 0.3),
                    "DOD_dir" (str): direction in which to compute DOD (default = "x"),
                    "fout" (str): path of the output csv file where to save the results (default = "volumes/res/dod/results.csv")
                }

    """
    default_cfg = {
        "grid_step": 0.3,
        "DOD_dir": "x",
        "fout": "volumes/res/dod/results.csv",
    }
    cfg = edict({**default_cfg, **{k: v for k, v in cfg.items()}})

    logger = logging.getLogger(current_process().name)
    logger.info(f"Processing {Path(pcd_pair[0]).stem}-{Path(pcd_pair[1]).stem} started")

    dod = DemOfDifference(pcd_pair)
    ret = dod.compute_volume(direction=cfg.DOD_dir, grid_step=cfg.grid_step)
    dod.write_result_to_file(cfg.fout, mode="a+", header=False)
    dod.clear()
    del dod
    gc.collect()
    logger.info(f"Completed.")

    return ret


if __name__ == "__main__":

    # Logger
    logger = logging.getLogger(current_process().name)

    # Paths
    pcd_dir = Path(PCD_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert pcd_dir.is_dir(), f"Directory '{pcd_dir}' does not exists."
    assert any(pcd_dir.iterdir()), f"Directory '{pcd_dir}' is empty."

    # Output file
    fout_name = (
        f"{PCD_PATTERN.split('*')[0]}_dir{DOD_DIR.upper()}_tstep{TSTEP}_grid{GRID_STEP}"
    )
    fout = out_dir / f"{fout_name}.csv"
    if fout.exists():
        logger.warning(f"Output file {fout} already exists. Removing it.")
        fout.unlink()

    # Build list of tuples with point cloud paths
    pcd_list = sorted(pcd_dir.glob(PCD_PATTERN))
    pairs, dates = make_pairs(pcd_list, TSTEP)
    pairs = [pair for pair in pairs.values()]

    # Config dictionary
    cfg = {
        "pcd_dir": PCD_DIR,
        "pcd_pattern": PCD_PATTERN,
        "out_dir": OUT_DIR,
        "DOD_dir": DOD_DIR,
        "t_step": TSTEP,
        "grid_step": GRID_STEP,
        "fout": str(fout),
    }
    with open(out_dir / f"{fout_name}_parameters.json", "w") as outfile:
        json.dump(cfg, outfile, indent=4)

    # Test task
    # ret = DOD_task(pairs[0], cfg)

    # Run task with multiprocessing
    logger.info("DOD computation started:")
    t0 = time.time()
    with Pool() as pool:
        results = pool.starmap(DOD_task, zip(pairs, repeat(cfg)))

    t1 = time.time()
    logger.info(f"DOD computation completed. Elapsed time: {t1-t0:.2f} sec")

    if any(np.invert(results)):
        for i, res in enumerate(results):
            if not res:
                print(f"Iteration {i} failed!")
