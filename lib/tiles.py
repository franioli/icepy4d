
import cv2
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import List, Union

# from lib.classes import Camera, Imageds
from classes import Camera, Imageds
from classes_new.images import Image


class Tiler:
    '''
    Class for dividing an image into tiles.
    '''

    def __init__(
        self,
        image: Image,
        grid: List[int] = [1, 1],
        overlap: int = 0,
        origin: List[int] = [0, 0],
    ) -> None:
        ''' Initialize class
        Parameters
        __________
        - image (Image):
        - grid (List[int], default=[1, 1]): List containing the number of rows and number of colums in which to divide the image ([nrows, ncols])
        - overlap (int, default=0): Number of pixel of overlap between adiacent tiles
        - origin (List[int], default=[0, 0]): List of coordinates [x,y] of the pixel from which the tiling starts (top-left corner of the first tile)
        __________
        Return: None
        '''

        self._image = image
        self._im_path = image.path
        self._w = int(self._image.width)
        self._h = int(self._image.height)
        self._origin = origin
        self._overlap = overlap
        self._nrow = grid[0]
        self._ncol = grid[1]
        self.limits = {}
        self.tiles = {}

    @property
    def grid(self) -> List:
        return [self._nrow, self._ncol]

    @property
    def tiles_limits(self) -> dict:
        return self.limits

    def compute_limits_by_grid(self) -> None:
        ''' Method to compute the limits of each tile (i.e. xmin,ymin,xmax,xmax), given the number or row and columns of the tile grid.

        Returns a dictionary containing the index of the tile (in row-major order, C-style) and a list of the bounding box coordinates as: 
        {0,[xmin, xmax, ymin, ymax]}
        {1,[xmin, xmax, ymin, ymax]}
        ....
        '''

        DX = round(
            (self._w - self._origin[0]) / self._ncol / 10
        ) * 10
        DY = round(
            (self._h - self._origin[1]) / self._nrow / 10
        ) * 10

        for col in range(self._ncol):
            for row in range(self._nrow):
                tile_idx = np.ravel_multi_index(
                    (row, col),
                    (self._nrow, self._ncol),
                    order='C'
                )
                xmin = max(self._origin[0], col*DX - self._overlap)
                ymin = max(self._origin[1], row*DY - self._overlap)
                xmax = xmin + DX + self._overlap - 1
                ymax = ymin + DY + self._overlap - 1
                self.limits[tile_idx] = (
                    xmin, ymin, xmax, ymax
                )

    def read_all_tiles(self) -> None:
        assert self._im_path is not None, 'Invalid image path'

        for idx, limit in self.limits.items():
            self.tiles[idx] = self._image.extract_patch(limit)

    def remove_tile(self, tile_idx) -> None:
        if tile_idx is None:
            self.tiles = {}
        else:
            self.tiles[idx] = []


if __name__ == '__main__':
    '''Test classes '''

    images = Imageds(Path('data/img2021/p0'))
    img = Image(images.get_image_path(0))

    tile_grid = (2, 1)
    origin = (1000, 0)

    tiles = Tiler(
        img,
        grid=tile_grid,
        origin=origin,
    )

    tiles.compute_limits_by_grid()
    tiles.read_all_tiles()

    for idx, tile in tiles.tiles.items():
        plt.subplot(tile_grid[0], tile_grid[1], idx+1)
        plt.imshow(tile)
    plt.show()
