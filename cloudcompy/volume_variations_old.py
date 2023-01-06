"""
Conda environment must be activated manually with:
source ~/CloudComPy/bin/condaCloud.sh activate CloudComPy310       
source ~/CloudComPy/bin/condaCloud.sh activate belpy_cc       

The script must be executed from the same terminal in which the conda environment was activated (do not try to run it with vscode or other IDE)

source ~/CloudComPy/bin/condaCloud.sh deactivate     

"""

import os
import gc
import psutil

# import linecache
# import tracemalloc
# import pdb

from pathlib import Path
from tqdm import tqdm
from typing import List, Union

# Import Cloud CloudComPy
# For debugging:
# os.environ["_CCTRACE_"] = "ON"  # only if you want debug traces from C++
# print(os.environ["PATH"])
# print(os.environ["PYTHONPATH"])
# print(os.environ["LD_LIBRARY_PATH"])
import cloudComPy as cc  # import the CloudComPy module

pcd_dir = Path("res/point_clouds")
i = 0
tstep = 3
verbose = False


class DOD:
    def __init__(self, pcd_list: List[str]) -> None:

        self.pcd0 = cc.loadPointCloud(pcd_list[0])
        self.pcd1 = cc.loadPointCloud(pcd_list[1])

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
        if verbose:
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


# def compute_volume_DOD(pcd_list: List[str]) -> None:

# pcd0 = cc.loadPointCloud(pcd_list[0])
# pcd1 = cc.loadPointCloud(pcd_list[1])
# # pdb.set_trace()

# # Compute variation by DOD: pcd(i) - pcd(i+step)
# if verbose:
#     print(
#         f"Computing volume variation between point clouds {str(pcd_list[0])} and {str(pcd_list[1])}"
#     )
# report = cc.ReportInfoVol()
# isOk = cc.ComputeVolume25D(
#     report,
#     ground=pcd0,
#     ceil=pcd1,
#     vertDim=0,
#     gridStep=0.20,
#     groundHeight=0,
#     ceilHeight=0,
# )
# if not isOk:
#     raise RuntimeError(
#         f"Unable to compute volume variation between point clouds {str(pcd_list[0])} and {str(pcd_list[1])}"
#     )

# if verbose:
#     print(
#         f"""Volume variation report:
#     Volume: {report.volume:.2f} m3
#     Added volume: {report.addedVolume:.2f} m3
#     Removed volume: {report.removedVolume:.2f} m3
#     Surface: {report.surface:.2f} m2
#     Maching Percent {report.matchingPercent:.1f}%
#     Average Neighbora per cell: {report.averageNeighborsPerCell:.1f}
#     """
#     )
# volume.append(report.volume)

# # snapshot = tracemalloc.take_snapshot()
# # display_top(snapshot)

# # Remove variables and free memory
# del pcd0
# del pcd1
# gc.collect()

# if verbose:
#     print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)


# Read point cloud list
assert pcd_dir.is_dir(), "Directory does not exists."
pcd_lst = sorted(pcd_dir.glob("dense_*.ply"))


pcd = cc.loadPointCloud(pcd_lst[0])


# Big Loop

# tracemalloc.start()
# process = psutil.Process(os.getpid())

# volume = []

# for i in tqdm(range(len(pcd_lst) - tstep)):
# compute_volume_DOD([str(pcd_lst[i]), str(pcd_lst[i + tstep])])

# for i in range(len(pcd_lst) - tstep):
#     # print(f"Iterazione {i} - ", end=" ")
#     print(f"Iterazione {i}")
#     print(
#         f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f} MB"
#     )

#     volum_diff = DOD([str(pcd_lst[i]), str(pcd_lst[i + tstep])])
#     volum_diff.compute_volume(direction="x", grid_step=0.20, verbose=False)
#     volume.append(volum_diff.report.volume)
#     print(f"estimated volume {volume[i]}")
#     del volum_diff
#     collected = gc.collect()
#     print(f"Garbage collector: collected {collected} objects.")

#     print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f} MB")
