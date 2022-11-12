import numpy as np
import cv2
import os

from pathlib import Path

from lib.classes import Features, Imageds

# @TODO MOVE ALL FUNCTIONS IN CORRECT import_export folder


def process_resize(w, h, resize):
    assert (len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    return w_new, h_new


def read_img(path, color=True, resize=[-1], crop=None):
    if color:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)

    if image is None:
        return None, None

    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    image = cv2.resize(image, (w_new, h_new))

    if crop:
        image = image[crop[1]:crop[3], crop[0]:crop[2]]

    return image, scales


'''
Export keypoints and points3d to file
'''


def export_keypoints_by_image(
    features: Features,
    imageds: Imageds,
    path: str = './',
    epoch: int = None,
) -> None:
    if epoch is not None:

        cams = list(imageds.keys())
        path = Path(path)

        for cam in cams:
            im_name = imageds[cam].get_image_stem(epoch)
            file = open(path / f'keypoints_{im_name}.txt', "w")

            # Write header to file
            file.write("feature_id,x,y\n")

            for id, kpt in enumerate(features[cam][epoch].get_keypoints()):
                x, y = kpt
                file.write(
                    f"{id},{x},{y}\n"
                )

        file.close()
        print("Marker exported successfully")
    else:
        print('please, provide the epoch number.')
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


'''
Export results to files
'''


def export_keypoints(
    filename: str,
    features: Features,
    imageds: Imageds,
    epoch: int = None,
) -> None:
    if epoch is not None:

        cams = list(imageds.keys())

        # Write header to file
        file = open(filename, "w")
        file.write("image_name, feature_id, x, y\n")

        for cam in cams:
            image_name = imageds[cam].get_image_name(epoch)

            # Write image name line
            # NB: must be manually modified if it contains characters of symbols
            file.write(f"{image_name}\n")

            for id, kpt in enumerate(features[cam][epoch].get_keypoints()):
                x, y = kpt
                file.write(
                    f"{id},{x},{y} \n"
                )

        file.close()
        print("Marker exported successfully")
    else:
        print('please, provide the epoch number.')
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


'''
Export data to file for CALGE
'''


def export_keypoints_for_calge(
    filename: str,
    features: Features,
    imageds: Imageds,
    epoch: int = None,
    pixel_size_micron: float = None,
) -> None:
    """ Write keypoints image coordinates to csv file,
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
        imageds (calsses.Imageds):
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
            image_name = imageds[cam].get_image_name(epoch)

            # Write image name line
            # NB: must be manually modified if it contains characters of symbols
            file.write(f"{image_name}\n")

            for id, kpt in enumerate(features[cam][epoch].get_keypoints()):
                x, y = kpt

                # If pixel_size_micron is not empty, convert image coordinates from x-y (row,column) image coordinate system to xi-eta image coordinate system (origin at the center of the image, xi towards right, eta upwards)
                if pixel_size_micron is not None:
                    xi = (x - img_size[1]/2) * pixel_size_micron
                    eta = (img_size[0]/2 - y) * pixel_size_micron

                    file.write(
                        f"{id:05}{xi:10.1f}{eta:15.1f} \n")
                else:
                    file.write(
                        f"{id:05}{x:10.1f}{y:15.1f} \n")
            # Write end image line
            file.write(f"-99\n")

        file.close()
        print("Marker exported successfully")
    else:
        print('please, provide the epoch number.')
        return


def export_points3D_for_calge(
    filename: str,
    points3D: np.ndarray,
) -> None:
    """ Write 3D world coordinates of matched points to csv file,
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
