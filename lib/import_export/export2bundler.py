import numpy as np
import open3d as o3d

from pathlib import Path
from copy import deepcopy
from shutil import copy as scopy
from typing import Union, List

from lib.base_classes.camera import Camera
from lib.base_classes.pointCloud import PointCloud
from lib.base_classes.features import Features
from lib.base_classes.images import Imageds
from lib.base_classes.targets import Targets
from lib.utils.utils import create_directory
from thirdparty.transformations import euler_from_matrix, euler_matrix


def write_bundler_out_single_epoch(
    export_dir: Union[str, Path],
    epoch: int,
    images: dict,
    cams: List[str],
    cameras: dict,
    features: dict,
    point_cloud: PointCloud,
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
    print("Exporting results in Bundler format...")
    export_dir = Path(export_dir)

    out_dir = create_directory(export_dir / "data")

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
    targets_enabled = [int(x) for x in targets_enabled]
    for i, target in enumerate(targets_to_use):
        for i, cam in enumerate(cams):
            for x in targets[epoch].extract_object_coor_by_label([target]).squeeze():
                file.write(f"{x:.4f} ")
            for x in (
                targets[epoch].extract_image_coor_by_label([target], cam_id=i).squeeze()
            ):
                file.write(f"{x:.4f} ")
            file.write(f"{images[cam].get_image_name(epoch)} ")
            file.write(f"{target} ")
            if len(targets_enabled) > 0:
                file.write(f"{targets_enabled[i]}\n")
    file.close()

    # Create Bundler output file
    num_cams = len(cams)
    num_pts = len(features[cams[0]][epoch])
    # w, h = 6012, 4008
    w = cameras[epoch][cam].width
    h = cameras[epoch][cam].height

    file = open(out_dir / f"belpy_epoch_{epoch}.out", "w")
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
    obj_coor = np.asarray(point_cloud.points)
    obj_col = (np.asarray(point_cloud.colors) * 255.0).astype(int)
    im_coor = {}
    for cam in cams:
        m = deepcopy(features[cam][epoch].get_keypoints())
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


def write_bundler_out(
    export_dir: Union[str, Path],
    epoches: List[int],
    images: dict,
    cams: List[str],
    cameras: dict,
    features: dict,
    # point_clouds: List[PointCloud],
    point_cloud: PointCloud,
    targets: List[Targets] = [],
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
    print("Exporting results in Bundler format...")
    export_dir = Path(export_dir)

    for epoch in epoches:
        out_dir = create_directory(export_dir / "data")

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
        targets_enabled = [int(x) for x in targets_enabled]
        for i, target in enumerate(targets_to_use):
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
                    file.write(f"{x+0.5:.4f} ")
                file.write(f"{images[cam].get_image_name(epoch)} ")
                file.write(f"{target} ")
                if len(targets_enabled) > 0:
                    file.write(f"{targets_enabled[i]}\n")
        file.close()

        # Create Bundler output file
        num_cams = len(cams)
        num_pts = len(features[cams[0]][epoch])
        w = cameras[epoch][cam].width
        h = cameras[epoch][cam].height

        file = open(out_dir / f"belpy_epoch_{epoch}.out", "w")
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
        obj_coor = deepcopy(point_cloud.get_points())
        obj_col = deepcopy(point_cloud.get_colors())
        # obj_coor = deepcopy(point_clouds[epoch].get_points())
        # obj_col = deepcopy(point_clouds[epoch].get_colors())
        im_coor = {}
        for cam in cams:
            m = deepcopy(features[cam][epoch].get_keypoints())
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

    print("Export completed.")


if __name__ == "main":
    pass
