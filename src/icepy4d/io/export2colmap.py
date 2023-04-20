import os
import numpy as np

from typing import Union
from pathlib import Path
from enum import Enum

import h5py
from collections import defaultdict
import torch
from copy import deepcopy

import icepy4d.classes as icepy4d_classes
from icepy4d.thirdparty.transformations import quaternion_from_matrix


class CameraModels(Enum):
    PINHOLE = 0
    RADIAL = 1
    OPENCV = 2
    FULL_OPENCV = 3


MIN_MATCHES = 20


def features_to_h5(
    features: icepy4d_classes.FeaturesDictEpoch, output_dir: Union[str, Path]
) -> bool:
    key1, key2 = images[cams[0]][epoch], images[cams[1]][epoch]

    cams = list(features.keys())

    mkpts0 = features[cams[0]].kpts_to_numpy()
    mkpts1 = features[cams[1]].kpts_to_numpy()
    n_matches = len(mkpts0)

    output_dir = Path(epochdir)
    db_name = output_dir / f"{epoch_dict[epoch]}.h5"
    with h5py.File(db_name, mode="w") as f_match:
        group = f_match.require_group(key1)
        if n_matches >= MIN_MATCHES:
            group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)

    with h5py.File(db_name, mode="r") as f_match:
        for k1 in f_match.keys():
            group = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0] += total_kpts[k1]
                current_match[:, 1] += total_kpts[k2]
                total_kpts[k1] += len(matches)
                total_kpts[k2] += len(matches)
                match_indexes[k1][k2] = current_match

    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(
            torch.from_numpy(kpts[k]), dim=0, return_inverse=True
        )
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
            out_match[k1][k2] = m2.numpy()
    with h5py.File(output_dir / f"keypoints.h5", mode="w") as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1

    with h5py.File(output_dir / f"matches.h5", mode="w") as f_match:
        for k1, gr in out_match.items():
            group = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match


def export_solution_to_colmap(
    export_dir: Union[str, Path],
    im_dict: dict,
    cameras: icepy4d_classes.CamerasDictEpoch,
    features: icepy4d_classes.FeaturesDictEpoch,
    points: icepy4d_classes.Points,
    camera_model: CameraModels = CameraModels.OPENCV,
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
        line = f"""{cam_id} {CameraModels.OPENCV} {cameras[cam].width} {cameras[cam].height} {" ".join(str(x) for x in params)}\n"""
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
