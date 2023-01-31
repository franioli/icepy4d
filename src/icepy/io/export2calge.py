import numpy as np
import logging


from ..classes.images import ImageDS
from ..classes.features import Features


"""
Export data to file for CALGE
"""


def export_keypoints_for_calge(
    filename: str,
    features: Features,
    imageds: ImageDS,
    epoch: int = None,
    pixel_size_micron: float = None,
) -> None:
    """Write keypoints image coordinates to csv file,
    sort by camera, as follows:
    cam1, kpt1, x, y
    cam1, kpt2, x, y
    ...
    cam1, kptM, x, y
    cam2, kpt1, x, y
    ....
    camN, kptM, x, y

    Args:
        filename (str): path of the output csv file
        features (calsses.Features):
        imageds (calsses.ImageDS):
        epoch (int, default = None):
        pixel_size_micron (float, default = None) [micron]
    """

    if epoch is not None:

        cams = list(imageds.keys())

        # Write header to file
        file = open(filename, "w")
        if pixel_size_micron is not None:
            file.write("image_name, feature_id, xi, eta\n")
            img = imageds[cams[0]][epoch]
            img_size = img.shape[:2]
        else:
            file.write("image_name, feature_id, x, y\n")

        for cam in cams:
            image_name = imageds[cam][epoch]

            # Write image name line
            # NB: must be manually modified if it contains characters of symbols
            file.write(f"{image_name}\n")

            for id, kpt in enumerate(features[epoch][cam].kpts_to_numpy()):
                x, y = kpt

                # If pixel_size_micron is not empty, convert image coordinates from x-y (row,column) image coordinate system to xi-eta image coordinate system (origin at the center of the image, xi towards right, eta upwards)
                if pixel_size_micron is not None:
                    xi = (x - img_size[1] / 2) * pixel_size_micron
                    eta = (img_size[0] / 2 - y) * pixel_size_micron

                    file.write(f"{id:05}{xi:10.1f}{eta:15.1f} \n")
                else:
                    file.write(f"{id:05}{x:10.1f}{y:15.1f} \n")
            # Write end image line
            file.write(f"-99\n")

        file.close()
        logging.info("Marker exported successfully")
    else:
        logging.error("please, provide the epoch number.")
        return


def export_points3D_for_calge(
    filename: str,
    points3D: np.ndarray,
) -> None:
    """Write 3D world coordinates of matched points to csv file,
    sort by camera, as follows:
    marker1, X, Y, Z
    ...
    markerM, X, Y, Z

    Args:
        filename (str): path of the output csv file
        points3D (np.ndarray):
    """

    # Write header to file
    file = open(filename, "w")
    file.write("point_id, X, Y, Z\n")

    for id, pt in enumerate(points3D):
        file.write(f"{id:05}{pt[0]:20.4f}{pt[1]:25.4f}{pt[2]:24.4f}\n")

    file.close()
    print("Points exported successfully")


if __name__ == "main":
    pass
