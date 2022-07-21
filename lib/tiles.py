from classes import Camera, Imageds
import os
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
# from collections import OrderedDict


class Tiles:

    def __init__(self, overlap):
        self.grid = []
        self.overlap = overlap
        self.origin = []
        self.tile = []
        self.limits = []

    def compute_grid(self,):
        print('method not implemented yet')

    def generate_tiles(self):
        print('method not implemented yet')

    def print_tiles(self):
        print('method not implemented yet')

    def write_tiles_to_disk(self):
        print('method not implemented yet')


def generateTiles(image, rowDivisor=2, colDivisor=2, overlap=200, viz=False, out_dir='tiles', writeTile2Disk=True):
    assert not (image is None), 'Invalid image input'

    image = image.astype('float32')
    H = image.shape[0]
    W = image.shape[1]
    DY = round(H/rowDivisor/10)*10
    DX = round(W/colDivisor/10)*10
    dim = (rowDivisor, colDivisor)

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


if __name__ == '__main__':
    '''Test classes '''

    import os

# from collections import OrderedDict

# from lib.classes import Camera, Imageds


class Tiles:

    def __init__(self,
                 im_path, im_shape=None,
                 nrows=1, ncols=1,
                 overlap=0, origin=[0, 0]):
        self.im_path = Path(im_path)
        self.im_shape = im_shape
        self.grid = {'nrows': nrows, 'ncols': ncols}
        self.overlap = overlap
        self.origin = origin
        self.tile_idx = []
        self.tiles = {}
        self.limits = {}

    def get_im_shape(self) -> None:
        ''' Read image to get image shape and save it to Class proprierty'''
        assert self.im_path is not None, 'Invalid image path'
        # assert self.im_path.exists()
        image = cv2.imread(str(self.im_path))
        h, w = image.shape[0:2]
        self.im_shape = (w, h)

    def compute_grid(
        self, im_shape=None,
        tile_grid=None, overlap=None,
    ) -> dict:
        ''' Funtion to compute the limits of each tile (i.e. xmin,ymin,xmax,xmax), given the number or row and columns of the tile grid.
        The tile grid can be given in a tuple (eg., (nrows, ncols)), or if None is give, the grid is taken from the self.grid proprierty.
        '''
        # @TODO: check why order of rows and cols in tile_gird is reversed...

        # Check if im_shape is provided
        # Otherwise read image and get image shape
        if im_shape is not None:
            shape = im_shape
        else:
            if self.im_shape is None:
                self.get_im_shape()
            shape = self.im_shape

        # Check if the tile_grid is given, otherwise reas it from self.grid.
        if tile_grid is not None:
            nrows, ncols = tile_grid
        else:
            nrows, ncols = self.grid['nrows'], self.grid['ncols']

        # Check if overlap is given , otherwise read it from self.overlap
        if overlap is None:
            overlap = self.overlap

        H = shape[0]
        W = shape[1]
        DY = round(H/nrows/10)*10
        DX = round(W/ncols/10)*10

        for col in range(ncols):
            for row in range(nrows):
                tile_idx = np.ravel_multi_index((row, col),
                                                (nrows, ncols), order='F'
                                                )
                self.tile_idx.append(tile_idx)

                xmin = max(0, col*DX - self.overlap)
                ymin = max(0, row*DY - self.overlap)
                xmax = xmin + DX+self.overlap - 1
                ymax = ymin + DY+self.overlap - 1
                self.limits[tile_idx] = (
                    xmin, ymin, xmax, ymax
                )

        return self.limits

    def read_all_tiles(self) -> None:
        assert self.im_path is not None, 'Invalid image path'

        image = cv2.imread(str(self.im_path))
        for name in self.tile_idx:
            self.tiles[name] = image[
                self.limits[name][0]:self.limits[name][2],
                self.limits[name][1]:self.limits[name][3],
            ]

    def purge_tile(self, tile_idx) -> None:
        if tile_idx is None:
            self.tiles = {}
        else:
            self.tiles[idx] = []

    def print_tiles(self):
        print('method not implemented yet')

    def write_tiles_to_disk(self):
        print('method not implemented yet')


def generateTiles(image, rowDivisor=2, colDivisor=2, overlap=200, viz=False, out_dir='tiles', writeTile2Disk=True):
    assert not (image is None), 'Invalid image input'

    image = image.astype('float32')
    H = image.shape[0]
    W = image.shape[1]
    DY = round(H/rowDivisor/10)*10
    DX = round(W/colDivisor/10)*10
    dim = (rowDivisor, colDivisor)

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


if __name__ == '__main__':
    '''Test classes '''

    from tiles import Tiles

    images = Imageds(Path('data/img/p0'))
    img = images[0]
    im_path = images.get_image_path(0)
    tiles = Tiles(im_path, (4000, 6000))
    # tiles.get_im_shape()

    tiles.compute_grid(tile_grid=(3, 2))
    print(tiles.limits)

    tiles.read_all_tiles()
    print('done')

    for idx in tiles.tile_idx:
        plt.subplot(2, 3, idx+1)
        plt.imshow(tiles.tiles[idx])
    plt.show()

    # print(tiles.im_shape)
#
