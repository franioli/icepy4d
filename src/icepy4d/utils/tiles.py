"""
MIT License

Copyright (c) 2022 Francesco Ioli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import logging

from pathlib import Path
from typing import List

from icepy4d.classes.images import Image, ImageDS


class Tiler:
    """
    Class for dividing an image into tiles.
    """

    def __init__(
        self,
        image: Image,
        grid: List[int] = [1, 1],
        overlap: int = 0,
        origin: List[int] = [0, 0],
    ) -> None:
        """Initialize class
        Parameters
        __________
        - image (Image):
        - grid (List[int], default=[1, 1]): List containing the number of rows and number of colums in which to divide the image ([nrows, ncols])
        - overlap (int, default=0): Number of pixel of overlap between adiacent tiles
        - origin (List[int], default=[0, 0]): List of coordinates [x,y] of the pixel from which the tiling starts (top-left corner of the first tile)
        __________
        Return: None
        """

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
        if not dict:
            return self.limits
        else:
            logging.warning(
                "Limits not available, compute them first with compute_limits_by_grid."
            )

    def compute_limits_by_grid(self) -> None:
        """Method to compute the limits of each tile (i.e. xmin,ymin,xmax,xmax), given the number or row and columns of the tile grid.

        Returns a dictionary containing the index of the tile (in row-major order, C-style) and a list of the bounding box coordinates as:
        {0,[xmin, xmax, ymin, ymax]}
        {1,[xmin, xmax, ymin, ymax]}
        ....
        """

        DX = round((self._w - self._origin[0]) / self._ncol / 10) * 10
        DY = round((self._h - self._origin[1]) / self._nrow / 10) * 10

        for col in range(self._ncol):
            for row in range(self._nrow):
                tile_idx = np.ravel_multi_index(
                    (row, col), (self._nrow, self._ncol), order="C"
                )
                xmin = max(self._origin[0], col * DX - self._overlap)
                ymin = max(self._origin[1], row * DY - self._overlap)
                xmax = xmin + DX + self._overlap - 1
                ymax = ymin + DY + self._overlap - 1
                self.limits[tile_idx] = (xmin, ymin, xmax, ymax)

    def read_all_tiles(self) -> None:
        """Read all tiles and store them in class instance"""
        assert self._im_path is not None, "Invalid image path"
        for idx, limit in self.limits.items():
            self.tiles[idx] = self._image.extract_patch(limit)

    def read_tile(self, idx) -> np.ndarray:
        """Extract tile given its idx (int) and return it"""
        assert self._im_path is not None, "Invalid image path"
        return self._image.extract_patch(self.limits[idx])

    def remove_tiles(self, tile_idx) -> None:
        if tile_idx is None:
            self.tiles = {}
        else:
            self.tiles[tile_idx] = []

    def display_tiles(self) -> None:
        for idx, tile in self.tiles.items():
            plt.subplot(tile_grid[0], tile_grid[1], idx + 1)
            plt.imshow(tile)
        plt.show()

    # def write_tiles_to_disk(self, ) -> None:
    #             isExist = os.path.exists(out_dir)
    #             if not isExist:
    #                 os.makedirs(out_dir)
    #             cv2.imwrite(
    #                 os.path.join(
    #                     out_dir,
    #                     "tile_"
    #                     + str(tileIdx)
    #                     + "_"
    #                     + str(limits[tileIdx][0])
    #                     + "_"
    #                     + str(limits[tileIdx][1])
    #                     + ".jpg",
    #                 ),
    #                 tile,
    #             )


"Old function"


def generateTiles(
    image,
    rowDivisor=2,
    colDivisor=2,
    overlap=200,
    viz=False,
    out_dir="tiles",
    writeTile2Disk=True,
):
    assert not (image is None), "Invalid image input"

    image = image.astype("float32")
    H = image.shape[0]
    W = image.shape[1]
    DY = round(H / rowDivisor / 10) * 10
    DX = round(W / colDivisor / 10) * 10
    dim = (rowDivisor, colDivisor)

    tiles = []
    limits = []
    for col in range(0, colDivisor):
        for row in range(0, rowDivisor):
            tileIdx = np.ravel_multi_index((row, col), dim, order="F")
            limits.append(
                (
                    max(0, col * DX - overlap),
                    max(0, row * DY - overlap),
                    max(0, col * DX - overlap) + DX + overlap,
                    max(0, row * DY - overlap) + DY + overlap,
                )
            )
            tile = image[
                limits[tileIdx][1] : limits[tileIdx][3],
                limits[tileIdx][0] : limits[tileIdx][2],
            ]
            tiles.append(tile)
            if writeTile2Disk:
                isExist = os.path.exists(out_dir)
                if not isExist:
                    os.makedirs(out_dir)
                cv2.imwrite(
                    os.path.join(
                        out_dir,
                        "tile_"
                        + str(tileIdx)
                        + "_"
                        + str(limits[tileIdx][0])
                        + "_"
                        + str(limits[tileIdx][1])
                        + ".jpg",
                    ),
                    tile,
                )

    return tiles, limits


if __name__ == "__main__":
    """Test classes"""

    images = ImageDS(Path("data/img/p1"))
    img = images.read_image(0)

    tile_grid = (2, 2)
    origin = (1000, 0)

    tiles = Tiler(
        img,
        grid=tile_grid,
        origin=origin,
    )

    tiles.compute_limits_by_grid()

    # t = tiles.read_tile(1)

    tiles.read_all_tiles()
    tiles.display_tiles()
