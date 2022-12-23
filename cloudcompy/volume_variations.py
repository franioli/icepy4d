import os
import gc
import psutil

import pandas as pd
from matplotlib import pyplot as plt

from pathlib import Path
from typing import List
from tqdm import tqdm

import cloudComPy as cc  # import the CloudComPy module


PCD_DIR = Path("res/point_clouds")
TSTEP = 3
VERBOSE = False
GRID_STEP = 0.2
DIR = "z"
FOUT = "cloudcompy/DOD_res_z_20cm.csv.csv"


class DOD:
    def __init__(self, pcd_list: List[str]) -> None:
        self.pcd0_name = pcd_list[0]
        self.pcd1_name = pcd_list[1]
        self.pcd0 = cc.loadPointCloud(self.pcd0_name)
        self.pcd1 = cc.loadPointCloud(self.pcd1_name)

    def compute_volume(
        self,
        direction: str = "z",
        grid_step: float = 1,
        verbose: bool = False,
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
                f"Unable to compute volume variation between point clouds {str(self.pcd_list[0])} and {str(self.pcd_list[1])}"
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
        cc.deleteEntity(self.pcd0)
        cc.deleteEntity(self.pcd1)

    def write_result_to_file(self, fname: str, mode="a+", header=True):
        """
        write_result_to_file _summary_

        Args:
            fname (str): _description_
            mode (str, optional): _description_. Defaults to "w".
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
                f"{self.pcd0_name},{self.pcd1_name},{self.report.volume:.4f},{self.report.addedVolume:.4f},{self.report.removedVolume:.4f},{self.report.surface:.4f},{self.report.matchingPercent:.1f}{self.report.averageNeighborsPerCell:.1f}\n"
            )

    @staticmethod
    def read_results_from_file(fname: str) -> pd.DataFrame:
        df = pd.read_csv(fname, sep=",", header=1)
        return df


assert PCD_DIR.is_dir(), "Directory does not exists."
pcd_lst = sorted(PCD_DIR.glob("dense*.ply"))

# Compute volumes
volume = []
for i in tqdm(range(len(pcd_lst) - TSTEP)):
    volum_diff = DOD([str(pcd_lst[i]), str(pcd_lst[i + TSTEP])])
    volum_diff.compute_volume(direction=DIR, grid_step=GRID_STEP)
    # volum_diff.print_result()
    volum_diff.write_result_to_file(FOUT, mode="a+")
    volume.append(volum_diff.report.volume)

    # Free memory and clear pointers
    volum_diff.clear()
    del volum_diff
    gc.collect()
    # print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f} MB")

# Read volume results from file and make plot

print("done.")

# df = DOD.read_results_from_file("volume_computation_res.csv")
# volume = df.iloc[:, 3].to_numpy()
plt.plot(volume)
