import logging
from copy import deepcopy
from pathlib import Path
from shutil import copy as scopy
from typing import List, Union, Dict

import numpy as np

from icepy4d.core import CamerasDict, FeaturesDict
from icepy4d.core.points import Points
from icepy4d.core.targets import Targets
from icepy4d.thirdparty.transformations import euler_matrix


def write_bundler_out(
    export_dir: Union[str, Path],
    im_dict: Dict[str, Path],
    cameras: CamerasDict,
    features: FeaturesDict,
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
    file = open(out_dir / "im_list.txt", "w")
    for cam in cams:
        file.write(f"{im_dict[cam]}\n")
    file.close()


    # Create Bundler output file
    num_cams = len(cams)
    num_pts = len(features[cams[0]])
    w = cameras[cam].width
    h = cameras[cam].height

    with open(out_dir / f"{date}.out", "w") as file:
        file.write(f"{num_cams} {num_pts}\n")

        # Write cameras
        for cam in cams:
            # Rotate the camera pose by 180 degrees around the x axis to match the BUndler coordinate system
            cam_ = deepcopy(cameras[cam])
            Rx = euler_matrix(np.pi, 0.0, 0.0)
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

    # Deprecated as the file im_list.txt contains the absolute paths to the images
    # Crates symbolic links to the images in subdirectory "data/images"
    # im_out_dir = out_dir / "images"
    # im_out_dir.mkdir(parents=True, exist_ok=True)
    # for cam in cams:
    #     src = im_dict[cam]
    #     dst = im_out_dir / im_dict[cam].name
    #     make_symlink(src, dst)

    # Write markers to file
    if targets is not None:
        assert (
            targets_to_use
        ), "Provide a list with the names of the targets to use as targets_to_use argument"
        if targets_enabled:
            assert len(targets_enabled) == len(
                targets_to_use
            ), "Invalid argument targets_enabled. Arguments targets_to_use and targets_enabled must have the same length."

        file = open(out_dir / "gcps.txt", "w")
        targets_enabled = [int(x) for x in targets_enabled]
        for i, target in enumerate(targets_to_use):
            for i, cam in enumerate(cams):
                # Try to read the target information. If some piece of information (i.e., image coords or objects coords) is missing (ValueError raised), skip the target and move to the next one
                try:
                    obj_coor = targets.get_object_coor_by_label([target])[0].squeeze()
                    im_coor = targets.get_image_coor_by_label([target], cam_id=i)[
                        0
                    ].squeeze()
                except ValueError:
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

    logging.info("Export to Bundler format completed.")


def read_bundler_file(file_path):
    cameras = []
    points_3d = []
    points_2d = []
    track_info = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_cameras, num_points = map(int, lines[0].split())

    current_line = 1
    for _ in range(num_cameras):
        camera_data = lines[current_line:current_line + 6]
        f, k1, k2 = map(float, camera_data[0].split())
        R = np.array([list(map(float, line.split())) for line in camera_data[1:4]])
        t = np.array(list(map(float, camera_data[4].split())))
        cameras.append({'f': f, 'k1': k1, 'k2': k2, 'R': R, 't': t})
        current_line += 6

    for _ in range(num_points):
        point_data = lines[current_line:current_line + 3]
        position = np.array(list(map(float, point_data[0].split())))
        color = np.array(list(map(int, point_data[1].split())))
        num_views = int(point_data[2])
        views = []
        for _ in range(num_views):
            view_data = lines[current_line + 3].split()
            camera_idx, key_idx, x, y = map(int, view_data)
            views.append({'camera_idx': camera_idx, 'key_idx': key_idx, 'x': x, 'y': y})
            current_line += 1
        points_3d.append(position)
        points_2d.append(views)
        current_line += 1

    return cameras, points_3d, points_2d
 

if __name__ == "__main__":
    
    file_path = "./sandbox/2022-05-01_14-01-15.out"
    cameras, points_3d, points_2d = read_bundler_file(file_path)
