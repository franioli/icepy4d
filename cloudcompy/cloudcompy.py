import pandas as pd

from pathlib import Path
from typing import List, Tuple

import cloudComPy as cc  # import the CloudComPy module


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
        ], "Invalid direction provided. Provide the name of the axis as a string. The following directions are allowed: ['x', 'y', 'z']"
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
        self.report = None

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
                f"{Path(self.pcd_pair[0]).stem},{Path(self.pcd_pair[1]).stem},{self.report.volume:.4f},{self.report.addedVolume:.4f},{self.report.removedVolume:.4f},{self.report.surface:.4f},{self.report.matchingPercent:.1f},{self.report.averageNeighborsPerCell:.1f}\n"
            )

    @staticmethod
    def read_results_from_file(
        fname: str, sep: str = ",", header: int = 0
    ) -> pd.DataFrame:
        df = pd.read_csv(fname, sep=sep, header=header)
        return df
