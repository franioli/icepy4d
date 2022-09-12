import numpy as np
import cv2
import os

from lib.classes import Features, Imageds
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')


def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
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


def generateTiles(image, rowDivisor=2, colDivisor=2, overlap=200, viz=False, out_dir='tiles', writeTile2Disk=True):
    assert not (image is None), 'Invalid image input'

    image = image.astype('float32')
    H = image.shape[0]
    W = image.shape[1]
    DY = round(H/rowDivisor/10)*10
    DX = round(W/colDivisor/10)*10
    dim = (rowDivisor, colDivisor)

    # TODO: implement checks on image dimension
    # Check image dimensions
    # if not W % colDivisor == 0:
    #     print('Number of columns non divisible by the ColDivisor. Removing last column.')
    #     image = image[:, 0:-1]
    # if not H % rowDivisor == 0:
    #     print('Number of rows non divisible by the RowDivisor. Removing last row')
    #     image = image[0:-1, :]

    tiles = []
    limits = []
    for col in range(0, colDivisor):
        for row in range(0, rowDivisor):
            tileIdx = np.ravel_multi_index((row, col), dim, order='F')
            limits.append((max(0, col*DX - overlap),
                           max(0, row*DY - overlap),
                           max(0, col*DX - overlap) + DX+overlap,
                           max(0, row*DY - overlap) + DY+overlap))
            # print(f'Tile {tileIdx}: xlim = ({ limits[tileIdx][0], limits[tileIdx][2]}), ylim = {limits[tileIdx][1], limits[tileIdx][3]}')
            tile = image[limits[tileIdx][1]:limits[tileIdx][3],
                         limits[tileIdx][0]:limits[tileIdx][2]]
            tiles.append(tile)
            if writeTile2Disk:
                isExist = os.path.exists(out_dir)
                if not isExist:
                    os.makedirs(out_dir)
                cv2.imwrite(os.path.join(out_dir, 'tile_'+str(tileIdx)+'_'
                                         + str(limits[tileIdx][0])+'_'+str(limits[tileIdx][1])+'.jpg'), tile)

    return tiles, limits

# Export data to file


def export_keypoints_for_calge(
    filename: str,
    features: Features,
    imageds: Imageds,
    epoch: int = None,
    pixel_size_micron: float = None,
) -> None:
    """ Write keypoints image coordinates to csv file,
    sort by camera, as follows:
    cam1, marker1, x, y
    cam1, marker2, x, y
    ...
    cam1, markerM, x, y
    cam2, marker1, x, y
    ....
    camN, markerM, x,Y

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

            for id, kpt in enumerate(features[cam][epoch].get_keypoints()):
                x, y = kpt

                # If pixel_size_micron is not empty, convert image coordinates from x-y (row,column) image coordinate system to xi-eta image coordinate system (origin at the center of the image, xi towards right, eta upwards)
                if pixel_size_micron is not None:
                    xi = (x - img_size[1]/2) * pixel_size_micron
                    eta = (img_size[0]/2 - y) * pixel_size_micron

                    file.write(
                        f"{image_name}, {id:05}, {xi:7.1f}, {eta:7.1f} \n")
                else:
                    file.write(
                        f"{image_name}, {id:05}, {x:.1f}, {y:.1f} \n")

        file.close()
        print("Marker exported successfully")
    else:
        print('please, provide the epoch number.')
        return


def export_points3d_for_calge(
    filename: str,
    features: Features,
    imageds: Imageds,
    epoch: int = None,
    pixel_size_micron: float = None,
) -> None:
    """ Write keypoints image coordinates to csv file,
    sort by camera, as follows:
    cam1, marker1, x, y
    cam1, marker2, x, y
    ...
    cam1, markerM, x, y
    cam2, marker1, x, y
    ....
    camN, markerM, x,Y

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

            for id, kpt in enumerate(features[cam][epoch].get_keypoints()):
                x, y = kpt

                # If pixel_size_micron is not empty, convert image coordinates from x-y (row,column) image coordinate system to xi-eta image coordinate system (origin at the center of the image, xi towards right, eta upwards)
                if pixel_size_micron is not None:
                    xi = (x - img_size[1]/2) * pixel_size_micron
                    eta = (img_size[0]/2 - y) * pixel_size_micron

                    file.write(
                        f"{image_name}, {id:05}, {xi:7.1f}, {eta:7.1f} \n")
                else:
                    file.write(
                        f"{image_name}, {id:05}, {x:.1f}, {y:.1f} \n")

        file.close()
        print("Marker exported successfully")
    else:
        print('please, provide the epoch number.')
        return
