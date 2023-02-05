import numpy as np
import os
import logging

from pathlib import Path
from copy import deepcopy
from shutil import copy as scopy
from typing import Union, List

from ..classes.point_cloud import PointCloud
from ..classes.points import Points
from ..classes.targets import Targets
from ..classes.typed_dict_classes import FeaturesDictEpoch, CamerasDictEpoch

from ..utils.utils import create_directory
from ..thirdparty.transformations import euler_from_matrix, euler_matrix


def write_bundler_out(
    export_dir: Union[str, Path],
    im_dict: dict,
    cameras: CamerasDictEpoch,
    features: FeaturesDictEpoch,
    points: Points,
    targets: Targets = None,
    targets_to_use: List[str] = [],
    targets_enabled: List[bool] = [],
) -> None:
    """
    Export solution in Bundler .out format.
    Refers to the official website for information about the .out format.
    https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6
    __________
    Parameters:
    -
    __________
    Return
        None
    """
    logging.info("Exporting results in Bundler format...")

    cams = list(cameras.keys())
    export_dir = Path(export_dir)
    date = export_dir.name
    out_dir = export_dir / "metashape" / "data"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write im_list.txt in the same directory
    file = open(out_dir / f"im_list.txt", "w")
    for cam in cams:
        file.write(f"{im_dict[cam]}\n")
    file.close()

    # Crates symbolic links to the images in subdirectory "data/images"
    im_out_dir = out_dir / "images"
    im_out_dir.mkdir(parents=True, exist_ok=True)
    for cam in cams:
        src = im_dict[cam]
        dst = im_out_dir / im_dict[cam].name
        if not dst.exists():
            os.symlink(src, dst)

    # Write markers to file
    if targets is not None:
        assert (
            targets_to_use
        ), "Provide a list with the names of the targets to use as targets_to_use argument"
        if targets_enabled:
            assert len(targets_enabled) == len(
                targets_to_use
            ), "Invalid argument targets_enabled. Arguments targets_to_use and targets_enabled must have the same length."

        file = open(out_dir / f"gcps.txt", "w")
        targets_enabled = [int(x) for x in targets_enabled]
        for i, target in enumerate(targets_to_use):
            for i, cam in enumerate(cams):
                # Try to read the target information. If some piece of information (i.e., image coords or objects coords) is missing (ValueError raised), skip the target and move to the next one
                try:
                    obj_coor = targets.get_object_coor_by_label([target])[0].squeeze()
                    im_coor = targets.get_image_coor_by_label([target], cam_id=i)[
                        0
                    ].squeeze()
                except ValueError as err:
                    logging.error(
                        f"Target {target} not found on image {im_dict[cam].name}. Skipped."
                    )
                    continue

                for x in obj_coor:
                    file.write(f"{x:.4f} ")
                for x in im_coor:
                    file.write(f"{x+0.5:.4f} ")
                file.write(f"{im_dict[cam].name} ")
                file.write(f"{target} ")
                if targets_enabled:
                    file.write(f"{targets_enabled[i]}\n")

        file.close()

    # Create Bundler output file
    num_cams = len(cams)
    num_pts = len(features[cams[0]])
    w = cameras[cam].width
    h = cameras[cam].height

    file = open(out_dir / f"{date}.out", "w")
    file.write(f"{num_cams} {num_pts}\n")

    # Write cameras
    Rx = euler_matrix(np.pi, 0.0, 0.0)
    for cam in cams:
        cam_ = deepcopy(cameras[cam])
        pose = cam_.pose @ Rx
        cam_.update_extrinsics(cam_.pose_to_extrinsics(pose))

        t = cam_.t.squeeze()
        R = cam_.R
        file.write(f"{cam_.K[1,1]:.10f} {cam_.dist[0]:.10f} {cam_.dist[1]:.10f}\n")
        for row in R:
            file.write(f"{row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n")
        file.write(f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f}\n")

    # Write points
    obj_coor = deepcopy(points.to_numpy())
    obj_col = deepcopy(points.colors_to_numpy(as_uint8=True))
    im_coor = {}
    for cam in cams:
        m = deepcopy(features[cam].kpts_to_numpy())
        # Convert image coordinates to bundler image rs
        m[:, 0] = m[:, 0] - w / 2
        m[:, 1] = h / 2 - m[:, 1]
        m = m + np.array([0.5, -0.5])
        im_coor[cam] = m

    for i in range(num_pts):
        file.write(f"{obj_coor[i][0]} {obj_coor[i][1]} {obj_coor[i][2]}\n")
        file.write(f"{obj_col[i][0]} {obj_col[i][1]} {obj_col[i][2]}\n")
        file.write(
            f"2 0 {i} {im_coor[cams[0]][i][0]:.4f} {im_coor[cams[0]][i][1]:.4f} 1 {i} {im_coor[cams[1]][i][0]:.4f} {im_coor[cams[1]][i][1]:.4f}\n"
        )

    file.close()

    logging.info("Export to Bundler format completed.")


def write_bundler_out_all_epoches(
    export_dir: Union[str, Path],
    epoches: List[int],
    images: dict,
    cams: List[str],
    cameras: dict,
    features: dict,
    point_clouds: List[PointCloud],
    targets: List[Targets] = [],
    targets_to_use: List[str] = [],
    targets_enabled: List[bool] = [],
) -> None:
    """
    Deprecated. Replaced by write_bundler_out function.

    Export solution in Bundler .out format.
    Refers to the official website for information about the .out format.
    https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6
    __________
    Parameters:
    -
    __________
    Return
        None
    """

    logging.info("Exporting results in Bundler format...")
    export_dir = Path(export_dir)

    for epoch in epoches:
        out_dir = create_directory(export_dir / "data")

        # Write im_list.txt in the same directory
        file = open(out_dir / f"im_list.txt", "w")
        for cam in cams:
            file.write(f"{images[cam][epoch]}\n")
        file.close()

        # Copy images in subdirectory "images"
        for cam in cams:
            im_out_dir = create_directory(out_dir / "images")
            scopy(
                images[cam].get_image_path(epoch),
                im_out_dir / images[cam][epoch],
            )

        # Write markers to file
        file = open(out_dir / f"gcps.txt", "w")
        targets_enabled = [int(x) for x in targets_enabled]
        for i, target in enumerate(targets_to_use):
            for i, cam in enumerate(cams):
                for x in targets[epoch].get_object_coor_by_label([target])[0].squeeze():
                    file.write(f"{x:.4f} ")
                for x in (
                    targets[epoch]
                    .get_image_coor_by_label([target], cam_id=i)[0]
                    .squeeze()
                ):
                    file.write(f"{x+0.5:.4f} ")
                file.write(f"{images[cam][epoch]} ")
                file.write(f"{target} ")
                if len(targets_enabled) > 0:
                    file.write(f"{targets_enabled[i]}\n")
        file.close()

        # Create Bundler output file
        num_cams = len(cams)
        num_pts = len(features[epoch][cams[0]])
        w = cameras[epoch][cam].width
        h = cameras[epoch][cam].height

        file = open(out_dir / f"icepy_epoch_{epoch}.out", "w")
        file.write(f"{num_cams} {num_pts}\n")

        # Write cameras
        Rx = euler_matrix(np.pi, 0.0, 0.0)
        for cam in cams:
            cam_ = deepcopy(cameras[epoch][cam])
            pose = cam_.pose @ Rx
            cam_.update_extrinsics(cam_.pose_to_extrinsics(pose))

            t = cam_.t.squeeze()
            R = cam_.R
            file.write(f"{cam_.K[1,1]:.10f} {cam_.dist[0]:.10f} {cam_.dist[1]:.10f}\n")
            for row in R:
                file.write(f"{row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n")
            file.write(f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f}\n")

        # Write points
        obj_coor = deepcopy(point_clouds[epoch].get_points())
        obj_col = deepcopy(point_clouds[epoch].get_colors())
        im_coor = {}
        for cam in cams:
            m = deepcopy(features[epoch][cam].kpts_to_numpy())
            # Convert image coordinates to bundler image rs
            m[:, 0] = m[:, 0] - w / 2
            m[:, 1] = h / 2 - m[:, 1]
            m = m + np.array([0.5, -0.5])
            im_coor[cam] = m

        for i in range(num_pts):
            file.write(f"{obj_coor[i][0]} {obj_coor[i][1]} {obj_coor[i][2]}\n")
            file.write(f"{obj_col[i][0]} {obj_col[i][1]} {obj_col[i][2]}\n")
            file.write(
                f"2 0 {i} {im_coor[cams[0]][i][0]:.4f} {im_coor[cams[0]][i][1]:.4f} 1 {i} {im_coor[cams[1]][i][0]:.4f} {im_coor[cams[1]][i][1]:.4f}\n"
            )

        file.close()

    logging.info("Export completed.")


if __name__ == "main":
    pass
