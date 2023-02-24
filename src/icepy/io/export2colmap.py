import os

from typing import Union
from pathlib import Path

import icepy.classes as icepy_classes
from icepy.thirdparty.transformations import quaternion_from_matrix


def export_solution_to_colmap(
    export_dir: Union[str, Path],
    im_dict: dict,
    cameras: icepy_classes.CamerasDictEpoch,
    features: icepy_classes.FeaturesDictEpoch,
    points: icepy_classes.Points,
    camera_model: str = "OPENCV",
) -> bool:

    cams = list(cameras.keys())
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Write cameras.txt
    file = open(export_dir / f"cameras.txt", "w")
    for cam_id, cam in enumerate(cams):
        # Using OPENCV camera model!
        params = [
            cameras[cam].K[0, 0],
            cameras[cam].K[1, 1],
            cameras[cam].K[0, 2],
            cameras[cam].K[1, 2],
            cameras[cam].dist[0],
            cameras[cam].dist[1],
            cameras[cam].dist[2],
            cameras[cam].dist[3],
        ]
        line = f"""{cam_id} {camera_model} {cameras[cam].width} {cameras[cam].height} {" ".join(str(x) for x in params)}\n"""
        file.write(line)
    file.close()

    # Write images.txt
    im_folder = export_dir / "images"
    file = open(export_dir / f"images.txt", "w")
    for cam_id, cam in enumerate(cams):
        quaternions = quaternion_from_matrix(cameras[cam].R)
        C = cameras[cam].C.squeeze()
        line = f"""{cam_id} {" ".join(str(x) for x in quaternions)} {" ".join(str(x) for x in C)} {cam_id} {im_folder.name}/{im_dict[cam].name}\n"""
        file.write(line)
        file.write("\n")
    file.close()

    # Crates symbolic links to the images in subdirectory "data/images"
    im_folder.mkdir(parents=True, exist_ok=True)
    for cam in cams:
        src = im_dict[cam]
        dst = im_folder / im_dict[cam].name
        if not dst.exists():
            os.symlink(src, dst)

    # Write empty points3D.txt
    file = open(export_dir / f"points3D.txt", "w")
    file.close()

    return True
