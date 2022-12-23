import sys
import gc

# a = "my"
# print(sys.getrefcount(a))

# b = [a]
# print(sys.getrefcount(a))

# c = {"key": a}  # Create a dictionary with a as one of the values.
# print(sys.getrefcount(a))


class MyClass(object):
    pass


print(gc.get_threshold())
# gc.collect()

dummy = MyClass()
dummy.obj = dummy
print(sys.getrefcount(dummy))

del dummy

print("done.")


import os
import gc
import psutil

from pathlib import Path
from typing import List

import cloudComPy as cc  # import the CloudComPy module


PCD_DIR = Path("...")
TSTEP = 1


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


assert PCD_DIR.is_dir(), "Directory does not exists."
pcd_lst = sorted(PCD_DIR.glob("*.ply"))

volume = []
for i in range(len(pcd_lst) - TSTEP):
    print(f"Iterazione {i}")
    print(
        f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f} MB"
    )

    volum_diff = DOD([str(pcd_lst[i]), str(pcd_lst[i + TSTEP])])
    volum_diff.compute_volume(direction="x", grid_step=0.20, verbose=False)
    volume.append(volum_diff.report.volume)
    print(f"estimated volume {volume[i]}")
    del volum_diff
    collected = gc.collect()
    print(f"Garbage collector: collected {collected} objects.")

    print(
        f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f} MB"
    )
