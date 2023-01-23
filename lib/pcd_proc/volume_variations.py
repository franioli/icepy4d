#%%
import gc
import time
import logging

import numpy as np
import pandas as pd
import open3d as o3d

from matplotlib import pyplot as plt
from pathlib import Path
from typing import List, Tuple, Union
from multiprocessing import Pool, current_process
from datetime import datetime, timedelta

from volumes.dem_of_difference import DemOfDifference
from volumes.utils import make_pairs

MP = False

PCD_DIR = "volumes/res/meshed"  # "volumes/data"  #
PCD_PATTERN = "sampled*.ply"  # "dense*.ply"  #
TSTEP = 5
VERBOSE = True
GRID_STEP = 0.3
DOD_DIR = "x"
OUT_DIR = "volumes/res/dod_x"
FOUT = "DOD_res_x_20cm.csv"

# polyline_path = "volumes/crop_polyline.poly"


LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)


def DOD_task(
    pcd_pair: Tuple,
) -> None:
    """
    DOD_task run DOD task for parallel computing

    Args:
        pcd_pair (Tuple): _description_
    """
    logger = logging.getLogger()
    process = current_process()
    logger.info(f"Child {process.name} - Epoch {Path(pcd_pair[0]).stem} started")

    dod = DemOfDifference(pcd_pair)
    dod.compute_volume(direction=DOD_DIR, grid_step=GRID_STEP)
    dod.write_result_to_file(Path(OUT_DIR) / FOUT, mode="a+", header=False)
    dod.clear()
    del dod
    gc.collect()
    logger.info(f"Child {process.name} completed.")


if __name__ == "__main__":

    pcd_dir = Path(PCD_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert pcd_dir.is_dir(), f"Directory '{pcd_dir}' does not exists."
    assert any(pcd_dir.iterdir()), f"Directory '{pcd_dir}' is empty."

    fout = out_dir / FOUT
    if fout.exists():
        logging.warning(f"Output file {fout} already exists. Removing it.")
        fout.unlink()

    pcd_list = sorted(pcd_dir.glob(PCD_PATTERN))
    pairs, dates = make_pairs(pcd_list, TSTEP)
    pairs = [pair for pair in pairs.values()]

    # Test task
    # DOD_task(pairs[0])

    logging.info("DOD computation started:")
    t0 = time.time()
    pool = Pool()
    pool.map(DOD_task, pairs)
    pool.close()
    t1 = time.time()
    logging.info(f"DOD computation completed. Elapsed timMaine: {t1-t0:.2f} sec")

    # Read volume results from file and make plot
    column_names = [
        "pcd0",
        "pcd1",
        "volume",
        "addedVolume",
        "removedVolume",
        "surface",
        "matchingPercent",
        "averageNeighborsPerCell",
    ]
    df = pd.read_csv(fout, sep=",", names=column_names)
    average_surface_match = df["matchingPercent"].to_numpy().mean()
    df["date_in"] = pd.to_datetime(
        df["pcd0"].str.replace(f"{PCD_PATTERN.split('*')[0]}_", ""), format="%Y_%m_%d"
    )
    df.sort_values(by="date_in", inplace=True)

    df["date_fin"] = pd.to_datetime(
        df["pcd1"].str.replace(f"{PCD_PATTERN.split('*')[0]}_", ""), format="%Y_%m_%d"
    )
    df["dt"] = (df.date_fin - df.date_in) / np.timedelta64(1, "D")
    df["volume_daily"] = df["volume"] / df["dt"]
    df["volume_daily_normalized"] = (
        df["volume_daily"] * df["matchingPercent"] / average_surface_match
    )

    df["volume_daily_cumul"] = df["volume_daily"].cumsum()
    df["volume_daily_norm_cumul"] = df["volume_daily_normalized"].cumsum()

    fname = f"{PCD_PATTERN.split('*')[0]}_step{TSTEP}_grid{GRID_STEP}.xlsx"
    df.to_excel(out_dir / fname, index=False)

    fig, ax = plt.subplots(1, 1)
    ax.grid(visible=True, which="both")
    ax.plot(df["date_in"], df["volume_daily"])
    ax.set_xlabel("day")
    ax.set_ylabel("m^3")
    ax.set_title("Daily volume difference")
    fig.set_size_inches(18.5, 10.5)
    fname = f"{PCD_PATTERN.split('*')[0]}_step{TSTEP}_grid{GRID_STEP}.png"
    fig.savefig(out_dir / fname, dpi=300)

    fig, ax = plt.subplots(1, 1)
    ax.grid(visible=True, which="both")
    ax.plot(df["date_in"], df["volume_daily_normalized"])
    ax.set_xlabel("day")
    ax.set_ylabel("m^3")
    ax.set_title("Daily volume difference")
    fig.set_size_inches(18.5, 10.5)
    fname = f"{PCD_PATTERN.split('*')[0]}_step{TSTEP}_grid{GRID_STEP}_normalized.png"
    fig.savefig(out_dir / fname, dpi=300)

    fig, ax = plt.subplots(1, 1)
    ax.grid(visible=True, which="both")
    ax.plot(df["date_in"], df["volume_daily_norm_cumul"])
    ax.set_xlabel("day")
    ax.set_ylabel("m^3")
    ax.set_title("Daily volume difference cumulated")
    fig.set_size_inches(18.5, 10.5)
    fname = f"{PCD_PATTERN.split('*')[0]}_step{TSTEP}_grid{GRID_STEP}_normalized_cumulated.png"
    fig.savefig(out_dir / fname, dpi=300)

    print(f"done")

    # # Single core processing

    # Read point cloud list
    # assert PCD_DIR.is_dir(), "Directory does not exists."
    # pcd_list = sorted(PCD_DIR.glob(PCD_PATTERN))
    # pairs = make_pairs(pcd_list, TSTEP)

    # pcd_dict = {}
    # pcd_dict['epoch'] = pcd_list[0].stem[9:11]
    # pcd_dict['date_ground'] = pcd_list[0].stem[12:]
    # pcd_dict = dict.fromkeys(list(map(lambda path: path.stem[9:11], pcd_list)))
    # pcd_dict = {k,v for k,v in enumerate(pcd_list)}

    # Big Loop: Compute volumes
    # results = {}
    # for iter, pair in tqdm(pairs.items()):
    #     dod = DOD(pair)
    #     dod.compute_volume(direction=DIR, grid_step=GRID_STEP)
    #     dod.write_result_to_file(FOUT, mode="a+")
    #     results[iter] = dod.report.volume
    #     if VERBOSE:
    #         dod.print_result()

    #     # Free memory and clear pointers
    #     dod.clear()
    #     del dod
    #     gc.colrelect()
    #     if VERBOSE:
    #         print(
    #             f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f} MB"
    #         )
    # print("done.")
