import numpy as np
import open3d as o3d

from pathlib import Path
from copy import deepcopy
from shutil import copy as scopy
from typing import Union, List

from lib.classes import Features, Imageds, Targets
from lib.utils import create_directory
from thirdparty.transformations import euler_from_matrix, euler_matrix


def write_bundler_out(
    export_dir: Union[str, Path],
    epoches: List[int],
    images: dict,
    cams: List[str],
    cameras: dict,
    features: dict,
    point_clouds: List[o3d.geometry.PointCloud],
    targets: List[Targets] = [],
    targets_to_use: List[str] = [],
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
    print("Exporting results in Bundler format...")

    for epoch in epoches:
        # for epoch in cfg.proc.epoch_to_process:
        # Output dir by epoch
        out_dir = create_directory(f"epoch_{epoch}/data")

        # Write im_list.txt in the same directory
        file = open(out_dir / f"im_list.txt", "w")
        for cam in cams:
            file.write(f"{images[cam].get_image_name(epoch)}\n")
        file.close()

        # Copy images in subdirectory "images"
        for cam in cams:
            im_out_dir = create_directory(out_dir / "images")
            scopy(
                images[cam].get_image_path(epoch),
                im_out_dir / images[cam].get_image_name(epoch),
            )

        # Write markers to file
        file = open(out_dir / f"gcps.txt", "w")
        for target in targets_to_use:
            for i, cam in enumerate(cams):
                for x in (
                    targets[epoch].extract_object_coor_by_label([target]).squeeze()
                ):
                    file.write(f"{x:.4f} ")
                for x in (
                    targets[epoch]
                    .extract_image_coor_by_label([target], cam_id=i)
                    .squeeze()
                ):
                    file.write(f"{x:.4f} ")
                file.write(f"{images[cam].get_image_name(epoch)} ")
                file.write(f"{target}\n")
        file.close()

        # Create Bundler output fileadd
        num_cams = len(cams)
        num_pts = len(features[cams[0]][epoch])
        w, h = 6012, 4008

        file = open(out_dir / f"belpy_epoch_{epoch}.out", "w")
        file.write(f"{num_cams} {num_pts}\n")

        # Write cameras
        Rx = euler_matrix(np.pi, 0.0, 0.0)
        for cam in cams:
            cam_ = deepcopy(cameras[cam][epoch])
            cam_.pose = cam_.pose @ Rx
            cam_.pose_to_extrinsics()

            t = cam_.t.squeeze()
            R = cam_.R
            file.write(f"{cam_.K[1,1]:.10f} {cam_.dist[0]:.10f} {cam_.dist[1]:.10f}\n")
            for row in R:
                file.write(f"{row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n")
            file.write(f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f}\n")

        # Write points
        obj_coor = np.asarray(point_clouds[epoch].points)
        obj_col = (np.asarray(point_clouds[epoch].colors) * 255.0).astype(int)
        im_coor = {}
        for cam in cams:
            m = features[cam][epoch].get_keypoints()
            m[:, 0] = m[:, 0] - w / 2
            m[:, 1] = h / 2 - m[:, 1]
            im_coor[cam] = m

        for i in range(num_pts):

            file.write(f"{obj_coor[i][0]} {obj_coor[i][1]} {obj_coor[i][2]}\n")
            file.write(f"{obj_col[i][0]} {obj_col[i][1]} {obj_col[i][2]}\n")
            file.write(
                f"2 0 {i} {im_coor[cams[0]][i][0]:.4f} {im_coor[cams[0]][i][1]:.4f} 1 {i} {im_coor[cams[1]][i][0]:.4f} {im_coor[cams[1]][i][1]:.4f}\n"
            )

        file.close()

    print("Export completed.")


if __name__ == "main":
    pass
