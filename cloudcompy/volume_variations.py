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

from cloudcompy import DOD

PCD_DIR = "cloudcompy/data"
PCD_PATTERN = "dense*.ply"
TSTEP = 3
VERBOSE = True
GRID_STEP = 0.2
DOD_DIR = "x"
OUT_DIR = "cloudcompy/res"
FOUT = "DOD_res_x_20cm.csv"  # "cloudcompy/DOD_res_x_20cm.csv"  #


def make_pairs(pcd_list: List[Path], step: int = 1) -> dict:
    pair_dict = {}
    for i in range(len(pcd_list) - step):
        pair_dict[i] = (str(pcd_list[i]), str(pcd_list[i + step]))
    return pair_dict


def DOD_task(
    pcd_pair: Tuple,
):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    process = current_process()
    logging.info(f"Child {process.name} - Epoch {Path(pcd_pair[0]).stem} started")

    dod = DOD(pcd_pair)
    dod.compute_volume(direction=DOD_DIR, grid_step=GRID_STEP)
    dod.write_result_to_file(Path(OUT_DIR) / FOUT, mode="a+", header=False)
    dod.clear()
    del dod
    gc.collect()
    logging.info(f"Child {process.name} completed.")


def meshing_task(
    pcd_path: Union[str, Path],
):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    process = current_process()
    logging.info(f"Child {process.name} - Processing pcd {pcd_path}.")

    pcd = o3d.io.read_point_cloud(pcd_path)
    assert Path(pcd).exists, "input point cloud does not exists."

    logging.info(f"Child {process.name} completed.")


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    pcd_dir = Path(PCD_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert pcd_dir.is_dir(), f"Directory '{pcd_dir}' does not exists."
    assert any(pcd_dir.iterdir()), f"Directory '{pcd_dir}' is empty."

    fout = out_dir / FOUT
    if fout.exists():
        logger.warning(f"Output file {fout} already exists. Removing it.")
        fout.unlink()

    pcd_list = sorted(pcd_dir.glob(PCD_PATTERN))
    pairs = make_pairs(pcd_list, TSTEP)
    pairs = [pair for pair in pairs.values()]

    logging.info("DOD computation started:")
    t0 = time.time()
    pool = Pool()
    pool.map(DOD_task, pairs)
    pool.close()
    t1 = time.time()
    logging.info(f"DOD computation completed. Elapsed timMaine: {t1-t0:.2f} sec")

    # # %% Read volume results from file and make plot
    # df = DOD.read_results_from_file(FOUT)
    # volumes = df.volume.to_numpy()

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(volumes)
    # ax.set_xlabel("day")
    # ax.set_ylabel("m^3")

    # print(f"done")

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

    # # Plot
    # volumes = list(results.values())
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(volumes)
    # ax.set_xlabel("epoch")
    # ax.set_ylabel("m^3")
    # ax.set_title(f"Volume difference with step of {TSTEP} days")
    # ax.grid(visible=True, color="k", linestyle="-", linewidth=0.2)
    # fig.savefig(Path(FOUT).parent / (Path(FOUT).stem + ".jpg"), dpi=300)
