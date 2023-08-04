import numpy as np
import logging
from pathlib import Path
from typing import Union

from icepy4d.thirdparty.transformations import euler_from_matrix
from icepy4d.classes.epoch import Epoch
from icepy4d.classes.features import Features
from icepy4d.classes.images import ImageDS


def write_cameras_to_disk(
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
