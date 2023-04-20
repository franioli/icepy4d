import numpy as np
import logging

from pathlib import Path

from ..classes.features import Features
from ..classes.images import ImageDS

"""
Export keypoints and points3d to file
"""


def export_keypoints(
    filename: str,
    features: Features,
    imageds: ImageDS,
    epoch: int = None,
) -> None:
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


def export_points3D(
    filename: str,
    points3D: np.ndarray,
) -> None:
    # Write header to file
    file = open(filename, "w")
    file.write("point_id,X,Y,Z\n")

    for id, pt in enumerate(points3D):
        file.write(f"{id},{pt[0]},{pt[1]},{pt[2]}\n")

    file.close()
    print("Points exported successfully")


if __name__ == "main":

    epoch = 0

    export_keypoints(
        "for_bba/keypoints_280722_for_bba.txt",
        features=features,
        imageds=images,
        epoch=epoch,
    )
    export_points3D(
        "for_bba/points3d_280722_for_bba.txt",
        points3D=np.asarray(point_clouds[epoch].points),
    )

    # Targets
    targets[epoch].im_coor[0].to_csv("for_bba/targets_p1.txt", index=False)
    targets[epoch].im_coor[1].to_csv("for_bba/targets_p2.txt", index=False)
    targets[epoch].obj_coor.to_csv("for_bba/targets_world.txt", index=False)
