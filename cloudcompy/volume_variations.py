#%%
import os
import gc
import psutil

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

import cloudComPy as cc  # import the CloudComPy module


PCD_DIR = Path("res/point_clouds")
PCD_PATTERN = "dense*.ply"
TSTEP = 3
VERBOSE = True
GRID_STEP = 0.2
DIR = "z"  # "x"  #
FOUT = "cloudcompy/DOD_res_x_20cm.csv"  # "cloudcompy/DOD_res_x_20cm.csv"  #


class DOD:
    def __init__(self, pcd_pair: Tuple[str]) -> None:
        self.pcd_pair = pcd_pair
        self.pcd0 = cc.loadPointCloud(self.pcd_pair[0])
        self.pcd1 = cc.loadPointCloud(self.pcd_pair[1])

    def compute_volume(
        self,
        direction: str = "x",
        grid_step: float = 1,
    ) -> None:
        assert direction in [
            "x",
            "y",
            "z",
        ], "Invalid direction provided. Provide the name of the axis as a string. The following directions are allowed: x, y, z"
        if direction == "x":
            direction = 0
        if direction == "y":
            direction = 1
        if direction == "z":
            direction = 2

        self.report = cc.ReportInfoVol()
        isOk = cc.ComputeVolume25D(
            self.report,
            ground=self.pcd0,
            ceil=self.pcd1,
            vertDim=direction,
            gridStep=grid_step,
            groundHeight=0,
            ceilHeight=0,
        )

        if not isOk:
            raise RuntimeError(
                f"Unable to compute volume variation between point clouds {str(self.pcd_pair[0])} and {str(self.pcd_pair[1])}"
            )

    def print_result(self) -> None:
        print(
            f"""Volume variation report:
            Volume: {self.report.volume:.2f} m3
            Added volume: {self.report.addedVolume:.2f} m3
            Removed volume: {self.report.removedVolume:.2f} m3
            Surface: {self.report.surface:.2f} m2
            Maching Percent {self.report.matchingPercent:.1f}%
            Average Neighbora per cell: {self.report.averageNeighborsPerCell:.1f}
            """
        )

    def clear(self):
        """
        clear Free memory occupied by the loaded point clouds
        """
        cc.deleteEntity(self.pcd0)
        cc.deleteEntity(self.pcd1)

    def write_result_to_file(self, fname: str, mode="a+", header=True):
        """
        write_result_to_file _summary_

        Args:
            fname (str): _description_
            mode (str, optional): _description_. Defaults to "a+".
        """

        if Path(fname).exists() and mode in ["a", "a+"]:
            write_header = False
        else:
            write_header = header

        with open(fname, mode=mode) as f:
            if write_header is True:
                # Write header
                f.write(
                    "pcd0,pcd1,volume,addedVolume,removedVolume,surface,matchingPercent,averageNeighborsPerCell\n"
                )
            f.write(
                f"{self.pcd_pair[0]},{self.pcd_pair[1]},{self.report.volume:.4f},{self.report.addedVolume:.4f},{self.report.removedVolume:.4f},{self.report.surface:.4f},{self.report.matchingPercent:.1f},{self.report.averageNeighborsPerCell:.1f}\n"
            )

    @staticmethod
    def read_results_from_file(
        fname: str, sep: str = ",", header: int = 0
    ) -> pd.DataFrame:
        df = pd.read_csv(fname, sep=sep, header=header)
        return df


def make_pairs(pcd_list: List[Path], step: int = 1) -> dict:
    pair_dict = {}
    for i in range(len(pcd_list) - step):
        pair_dict[i] = (str(pcd_list[i]), str(pcd_list[i + step]))
    return pair_dict


#%%

# Read point cloud list
assert PCD_DIR.is_dir(), "Directory does not exists."
pcd_list = sorted(PCD_DIR.glob(PCD_PATTERN))
pairs = make_pairs(pcd_list, TSTEP)

# pcd_dict = {}
# pcd_dict['epoch'] = pcd_list[0].stem[9:11]
# pcd_dict['date_ground'] = pcd_list[0].stem[12:]
# pcd_dict = dict.fromkeys(list(map(lambda path: path.stem[9:11], pcd_list)))
# pcd_dict = {k,v for k,v in enumerate(pcd_list)}

# Big Loop: Compute volumes
results = {}
for iter, pair in tqdm(pairs.items()):
    dod = DOD(pair)
    dod.compute_volume(direction=DIR, grid_step=GRID_STEP)
    dod.write_result_to_file(FOUT, mode="a+")
    results[iter] = dod.report.volume
    if VERBOSE:
        dod.print_result()

    # Free memory and clear pointers
    dod.clear()
    del dod
    gc.collect()
    if VERBOSE:
        print(
            f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f} MB"
        )
print("done.")

# Plot
volumes = list(results.values())
fig, ax = plt.subplots(1, 1)
ax.plot(volumes)
ax.set_xlabel("epoch")
ax.set_ylabel("m^3")
ax.set_title(f"Volume difference with step of {TSTEP} days")
ax.grid(visible=True, color="k", linestyle="-", linewidth=0.2)
fig.savefig(Path(FOUT).parent / (Path(FOUT).stem + ".jpg"), dpi=300)

#%% With Parallel computing
# from multiprocessing import Process
# from multiprocessing import Pool

# def do_iter(
#     iter: int,
#     pcd_pair: Tuple,
#     # out_dict: dict,
#     # step: int = TSTEP,
#     # direction: str = DIR,
#     # grid_step: float = GRID_STEP,
# ):
#     dod = DOD(pcd_pair)
#     dod.compute_volume(direction=DIR, grid_step=GRID_STEP)
#     dod.write_result_to_file(FOUT, mode="a+")
#     dod.print_result()
#     volume[iter] = dod.report.volume
#     dod.clear()
#     del dod
#     gc.collect()


# volume = {}
# pairs = make_pairs(pcd_list, TSTEP)
# # res = map(do_iter, pairs)

# p = Pool(4)

#%% Read volume results from file and make plot
# df = DOD.read_results_from_file(FOUT)
# volumes = df.volume.to_numpy()

# fig, ax = plt.subplots(1, 1)
# ax.plot(volumes)
# ax.set_xlabel("day")
# ax.set_ylabel("m^3")
