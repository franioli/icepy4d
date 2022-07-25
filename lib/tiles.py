from classes import Camera, Imageds
import os
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from classes import Imageds


class Tiles:

    def __init__(self,
                 im_path, im_shape=None,
                 nrows=1, ncols=1,
                 overlap=0, origin=(0, 0),
                 ):
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
        self.im_shape = (h, w)

    def compute_grid(self,
                     tile_grid=None, im_shape=None,
                     origin=(0, 0), overlap=None,
                     ) -> dict:
        ''' Funtion to compute the limits of each tile (i.e. xmin,ymin,xmax,xmax), given the number or row and columns of the tile grid.
        The tile grid can be given in a tuple (eg., (nrows, ncols)), or if None is give, the grid is taken from the self.grid proprierty.

        Note: array of tiles are arranged in row-major order
        '''
        # @TODO: check why order of rows and cols in tile_gird is reversed...

        # Check if im_shape is provided
        # Otherwise read image and get image shape
        if im_shape is not None:
            W, H = im_shape
        else:
            if self.im_shape is None:
                self.get_im_shape()
            H, W = self.im_shape

        # Check if the tile_grid is given, otherwise reas it from self.grid.
        if tile_grid is not None:
            nrows, ncols = tile_grid
        else:
            nrows, ncols = self.grid['nrows'], self.grid['ncols']

        # Check if overlap is given, otherwise read it from self.overlap
        if overlap is None:
            overlap = self.overlap

        if origin is None:
            origin = self.origin

        DX = round((W-origin[0])/ncols/10)*10
        DY = round((H-origin[1])/nrows/10)*10

        for col in range(ncols):
            for row in range(nrows):
                tile_idx = np.ravel_multi_index((row, col),
                                                (nrows, ncols), order='F'
                                                )
                self.tile_idx.append(tile_idx)

                xmin = max(origin[0], col*DX - self.overlap)
                ymin = max(origin[1], row*DY - self.overlap)
                xmax = xmin + DX+self.overlap - 1
                ymax = ymin + DY+self.overlap - 1
                self.limits[tile_idx] = (
                    xmin, ymin, xmax, ymax
                )

        return self.limits

    def read_all_tiles(self) -> None:
        assert self.im_path is not None, 'Invalid image path'

        image = cv2.imread(str(self.im_path))
        for idx in self.tile_idx:
            self.tiles[idx] = image[
                self.limits[idx][1]:self.limits[idx][3],
                self.limits[idx][0]:self.limits[idx][2],
            ]

    def purge_tile(self, tile_idx) -> None:
        if tile_idx is None:
            self.tiles = {}
        else:
            self.tiles[idx] = []

    def print_tiles(self):
        print('method not implemented yet')

        # for idx in tiles.tile_idx:
        #     plt.subplot(tile_grid[0], tile_grid[1], idx+1)
        #     plt.imshow(tiles.tiles[idx])
        # plt.show()

        # fig, ax = plt.subplots(tile_grid[0], tile_grid[1])
        # for idx in tiles.tile_idx:
        #     i, j = np.unravel_index(idx, tile_grid, order='F')
        #     ax[i][j].imshow(tiles.tiles[idx])
        # fig.show()

    def write_tiles_to_disk(self):
        print('method not implemented yet')


# def generateTiles(image, rowDivisor=2, colDivisor=2, overlap=200, viz=False, out_dir='tiles', writeTile2Disk=True):
#     assert not (image is None), 'Invalid image input'

#     image = image.astype('float32')
#     H = image.shape[0]
#     W = image.shape[1]
#     DY = round(H/rowDivisor/10)*10
#     DX = round(W/colDivisor/10)*10
#     dim = (rowDivisor, colDivisor)

#     tiles = []
#     limits = []
#     for col in range(0, colDivisor):
#         for row in range(0, rowDivisor):
#             tileIdx = np.ravel_multi_index((row, col), dim, order='C')
#             limits.append((max(0, col*DX - overlap),
#                            max(0, row*DY - overlap),
#                            max(0, col*DX - overlap) + DX+overlap,
#                            max(0, row*DY - overlap) + DY+overlap))
#             # print(f'Tile {tileIdx}: xlim = ({ limits[tileIdx][0], limits[tileIdx][2]}), ylim = {limits[tileIdx][1], limits[tileIdx][3]}')
#             tile = image[limits[tileIdx][1]:limits[tileIdx][3],
#                          limits[tileIdx][0]:limits[tileIdx][2]]
#             tiles.append(tile)
#             if writeTile2Disk:
#                 isExist = os.path.exists(out_dir)
#                 if not isExist:
#                     os.makedirs(out_dir)
#                 cv2.imwrite(os.path.join(out_dir, 'tile_'+str(tileIdx)+'_'
#                                          + str(limits[tileIdx][0])+'_'+str(limits[tileIdx][1])+'.jpg'), tile)

#     return tiles, limits


if __name__ == '__main__':
    '''Test classes '''

    from tiles import Tiles

    images = Imageds(Path('data/img/p0'))
    img = images[0]
    im_path = images.get_image_path(0)
    tiles = Tiles(im_path,
                  #   (4000, 6000),
                  )
    tiles.get_im_shape()

    tile_grid = (1, 2)
    origin = (1000, 0)
    im_shape = (4000, 4000)
    tiles.compute_grid(tile_grid=tile_grid,
                       im_shape=im_shape,
                       origin=origin,
                       )
    print(tiles.limits)
    tiles.read_all_tiles()

    for idx in tiles.tile_idx:
        plt.subplot(tile_grid[0], tile_grid[1], idx+1)
        plt.imshow(tiles.tiles[idx])
    plt.show()
    
    
    