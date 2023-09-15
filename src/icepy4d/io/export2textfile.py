import numpy as np
import logging
from pathlib import Path
from typing import Union
import pandas as pd

from icepy4d.thirdparty.transformations import euler_from_matrix
from icepy4d.core.epoch import Epoch
from icepy4d.core.features import Features
from icepy4d.core.images import ImageDS


def write_cameras_to_file(
    output_path: Union[Path, str], epoch: Epoch, sep: str = ","
) -> None:
    """Write camera parameters for a given epoch to a CSV file.

    Args:
        output_path (Union[Path, str]): The path to the output CSV file.
        epoch (Epoch): The Epoch object containing camera parameters.
        sep (str, optional): The separator to use in the CSV file. Defaults to ",".

    Returns:
        None: The function does not return anything.

    """
    if not isinstance(epoch, Epoch):
        logging.error("Invalid epoch object.")
        return

    output_path = Path(output_path)
    if not output_path.exists():
        items = [
            "date",
            "f1",
            "omega1",
            "phi1",
            "kappa1",
            "f2",
            "omega2",
            "phi2",
            "kappa2",
        ]
        with open(output_path, "w") as file:
            file.write(f"{f'{sep}'.join(items)}\n")

    with open(output_path, "a") as file:
        file.write(f"{str(epoch.timestamp)}")
        for cam in epoch.cameras.keys():
            f = epoch.cameras[cam].K[1, 1]
            # R = epoch.cameras[cam].R
            R = epoch.cameras[cam].pose[:3, :3]
            o, p, k = euler_from_matrix(R)
            o, p, k = np.rad2deg(o), np.rad2deg(p), np.rad2deg(k)
            file.write(f"{sep}{f:.2f}{sep}{o:.4f}{sep}{p:.4f}{sep}{k:.4f}")
        file.write("\n")


def write_reprojection_error_to_file(
    output_path: Union[Path, str], epoch: Epoch, sep: str = ","
) -> None:
    """Write reprojection error statistics to a CSV file.

    This function computes and writes the reprojection error statistics for
    each camera in the epoch to a CSV file. The reprojection error is the
    difference between the projected 3D points and their corresponding 2D
    keypoints in each camera. The statistics include the mean, standard
    deviation, minimum, 25th percentile, median, 75th percentile, and maximum
    of the reprojection errors for each camera and a global norm computed as
    the mean of all cameras' reprojection errors.

    Args:
        output_path (Union[Path, str]): The path to the output CSV file.
        epoch (Epoch): The Epoch object containing camera parameters and
        keypoints.
        sep (str, optional): The separator to use in the CSV file. Defaults to
        ",".

    Returns:
        None: The function does not return anything. The statistics are written
        to the CSV file.

    """

    cams = list(epoch.cameras.keys())
    output_path = Path(output_path)

    residuals = pd.DataFrame()
    for cam_key, camera in epoch.cameras.items():
        feat = epoch.features[cam_key]
        projections = camera.project_point(epoch.points.to_numpy())
        res = projections - feat.kpts_to_numpy()
        res_norm = np.linalg.norm(res, axis=1)
        residuals["track_id"] = feat.get_track_ids()
        residuals[f"x_{cam_key}"] = res[:, 0]
        residuals[f"y_{cam_key}"] = res[:, 1]
        residuals[f"norm_{cam_key}"] = res_norm

    # Compute global norm as mean of all cameras
    residuals["global_norm"] = np.mean(
        residuals[[f"norm_{x}" for x in cams]].to_numpy(), axis=1
    )
    res_stas = residuals.describe()
    res_stas_s = res_stas.stack()

    if not output_path.exists():
        with open(output_path, "w") as f:
            header_line = (
                "ep"
                + sep
                + f"{sep}".join([f"{x[0]}-{x[1]}" for x in res_stas_s.index.to_list()])
            )
            f.write(header_line + "\n")
    with open(output_path, "a") as f:
        line = (
            str(epoch.timestamp)
            + sep
            + f"{sep}".join([str(x) for x in res_stas_s.to_list()])
        )
        f.write(line + "\n")


"""
Export keypoints and points3d to file
"""


def export_keypoints(
    filename: str,
    features: Features,
    imageds: ImageDS,
    epoch: int = None,
) -> None:
    """Export keypoints for a given epoch and image dataset to a CSV file.

    Args:
        filename (str): The name of the output CSV file.
        features (Features): The Features object containing keypoints.
        imageds (ImageDS): The ImageDS object containing image data.
        epoch (int, optional): The epoch number to export keypoints. Defaults to None.

    Returns:
        None: The function does not return anything.

    """
    if epoch is not None:
        cams = list(imageds.keys())

        # Write header to file
        file = open(filename, "w")
        file.write("image_name, feature_id, x, y\n")

        for cam in cams:
            image_name = imageds[cam][epoch]

            # Write image name line
            # NB: must be manually modified if it contains characters of symbols
            file.write(f"{image_name}\n")

            for id, kpt in enumerate(features[epoch][cam].kpts_to_numpy()):
                x, y = kpt
                file.write(f"{id},{x},{y} \n")

        file.close()
        logging.info("Marker exported successfully")
    else:
        logging.error("please, provide the epoch number.")
        return


def export_points3D(
    filename: str,
    points3D: np.ndarray,
) -> None:
    """Export 3D points to a CSV file.

    Args:
        filename (str): The name of the output CSV file.
        points3D (np.ndarray): The numpy array containing 3D points.

    Returns:
        None: The function does not return anything.

    """
    # Write header to file
    file = open(filename, "w")
    file.write("point_id, X, Y, Z\n")

    for id, pt in enumerate(points3D):
        file.write(f"{id},{pt[0]},{pt[1]},{pt[2]}\n")

    file.close()
    print("Points exported successfully")


def export_keypoints_by_image(
    features: Features,
    imageds: ImageDS,
    path: str = "./",
    epoch: int = None,
) -> None:
    """Export keypoints for a given epoch and image dataset to separate CSV files
    for each camera image.

    Args:
        features (Features): The Features object containing keypoints.
        imageds (ImageDS): The ImageDS object containing image data.
        path (str, optional): The output path for the CSV files. Defaults to "./".
        epoch (int, optional): The epoch number to export keypoints. Defaults to None.

    Returns:
        None: The function does not return anything.

    """
    if epoch is not None:
        cams = list(imageds.keys())
        path = Path(path)

        for cam in cams:
            im_name = imageds[cam].get_image_stem(epoch)
            file = open(path / f"keypoints_{im_name}.txt", "w")

            # Write header to file
            file.write("feature_id,x,y\n")

            for id, kpt in enumerate(features[epoch][cam].kpts_to_numpy()):
                x, y = kpt
                file.write(f"{id},{x},{y}\n")

        file.close()
        print("Marker exported successfully")
    else:
        print("please, provide the epoch number.")
        return


if __name__ == "main":
    pass
    # export_keypoints(
    #     "for_bba/keypoints_280722_for_bba.txt",
    #     features=features,
    #     imageds=images,
    #     epoch=epoch,
    # )
    # export_points3D(
    #     "for_bba/points3d_280722_for_bba.txt",
    #     points3D=np.asarray(point_clouds[epoch].points),
    # )

    # # Targets
    # targets[epoch].im_coor[0].to_csv("for_bba/targets_p1.txt", index=False)
    # targets[epoch].im_coor[1].to_csv("for_bba/targets_p2.txt", index=False)
    # targets[epoch].obj_coor.to_csv("for_bba/targets_world.txt", index=False)
