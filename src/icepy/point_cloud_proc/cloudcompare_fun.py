from pathlib import Path
from typing import List, Tuple, Union
from pathlib import Path

import cloudComPy as cc  # import the CloudComPy module

ALLOWED_PCD_EXT = [".asc", ".las", ".E57", ".ply", ".pcd", ".bin"]


def cut_point_cloud_by_polyline(
    pcd: cc.ccPointCloud,
    polyline_path: str,
    direction: str = "z",
    inside: bool = True,
    output_pah: Union[str, Path] = None,
    delete_original: bool = False,
) -> cc.ccPointCloud:
    """
    cut_point_cloud_by_polyline - Function to crop a point cloud using a polyline.

    NOTE:
        Function is currently not working!

    Args:
        pcd (cc.ccPointCloud): Point cloud object to be cropped.
        polyline_path (str): Path to polyline file.
        direction (str, optional): Direction along which to perform cropping, can be "x", "y" or "z". Defaults to "z".
        inside (bool, optional): Flag to control cropping within (True) or outside (False) the polyline region. Defaults to True.
        output_pah (Union[str, Path], optional): Path to output cropped point cloud file. Defaults to None.
        delete_original (bool, optional): Flag to control whether to delete the original point cloud object. Defaults to False.

    Raises:
        RuntimeError: If unable to crop point cloud.
        IOError: If unable to save cropped point cloud to specified file path.

    Returns:
        cc.ccPointCloud: Cropped point cloud object.
    """
    assert direction in [
        "x",
        "y",
        "z",
    ], "Invalid direction provided. Provide the name of the axis as a string. The following directions are allowed: ['x', 'y', 'z']"
    if direction == "y":
        direction = 0
    if direction == "x":
        direction = 1
    if direction == "z":
        direction = 2

    polyline = cc.loadPolyline(polyline_path)
    polyline.setClosed(True)

    cropped = pcd.crop2D(polyline, direction, inside)
    if cropped is None:
        raise RuntimeError("Unable to crop point cloud.")

    if output_pah := Path(output_pah):
        assert (
            output_pah.suffix in ALLOWED_PCD_EXT
        ), f"Invalid point cloud extension. It must be one of the followings {ALLOWED_PCD_EXT}"
        output_pah.parent.mkdir(exist_ok=True, parents=True)
        if not cc.SavePointCloud(cropped, str(output_pah)):
            raise IOError(f"Unable to save cropped point cloud to {output_pah}.")

    # Free memory by deleting original CC entities
    cc.deleteEntity(polyline)
    if delete_original:
        cc.deleteEntity(pcd)

    return cropped


class DemOfDifference:
    def __init__(self, pcd_pair: Tuple[str]) -> None:
        self.pcd_pair = pcd_pair
        self.pcd0 = cc.loadPointCloud(self.pcd_pair[0])
        if self.pcd0 is None:
            raise IOError(f"Unable to read point cloud {self.pcd_pair[0]}")
        self.pcd1 = cc.loadPointCloud(self.pcd_pair[1])
        if self.pcd1 is None:
            raise IOError(f"Unable to read point cloud {self.pcd_pair[1]}")

    def compute_volume(
        self,
        direction: str = "x",
        grid_step: float = 1,
    ) -> bool:
        assert direction in [
            "x",
            "y",
            "z",
        ], "Invalid direction provided. Provide the name of the axis as a string. The following directions are allowed: ['x', 'y', 'z']"
        if direction == "x":
            self.direction = 0
        if direction == "y":
            self.direction = 1
        if direction == "z":
            self.direction = 2

        self.report = cc.ReportInfoVol()
        isOk = cc.ComputeVolume25D(
            self.report,
            ground=self.pcd0,
            ceil=self.pcd1,
            vertDim=self.direction,
            gridStep=grid_step,
            groundHeight=0,
            ceilHeight=0,
        )

        if not isOk:
            raise RuntimeError(
                f"Unable to compute volume variation between point clouds {str(self.pcd_pair[0])} and {str(self.pcd_pair[1])}"
            )
        else:
            return True

    def cut_point_clouds_by_polyline(
        self, polyline_path: str, direction: str = "x"
    ) -> None:
        """
        cut_point_clouds_by_polyline Function is currently not working!

        TODO:
            find bug to make function cut_point_cloud_by_polyline() working

        Args:
            polyline_path (str): _description_
            direction (str, optional): _description_. Defaults to "x".
        """
        self.pcd0 = cut_point_cloud_by_polyline(
            self.pcd0, polyline_path, direction, delete_original=True
        )
        self.pcd1 = cut_point_cloud_by_polyline(
            self.pcd1, polyline_path, direction, delete_original=True
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

    # @staticmethod
    # def read_results_from_file(
    #     fname: str,
    #     sep: str = ",",
    #     header: int = None,
    #     column_names: List[str] = None,
    # ) -> pd.DataFrame:

    #     if column_names is not None:
    #         df = pd.read_csv(fname, sep=sep, names=column_names)
    #     elif header is not None:
    #         df = pd.read_csv(fname, sep=sep, header=header)
    #     else:
    #         df = pd.read_csv(fname, sep=sep)

    #     return df


if __name__ == "__main__":

    pcd_path = "test/dense_2022_06_02.ply"
    polyline_path = "test/poly.poly"

    output_dir = "test"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    pcd = cc.loadPointCloud(pcd_path)

    out_path = output_dir / "test.ply"
    cropped = cut_point_cloud_by_polyline(
        pcd, polyline_path, direction="x", output_pah=out_path
    )
