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

# TODO: use KORNIA for image tiling


import logging
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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
        """
        Initialize class.

        Parameters:
        - image (Image): The input image.
        - grid (List[int], default=[1, 1]): List containing the number of rows and number of columns in which to divide the image ([nrows, ncols]).
        - overlap (int, default=0): Number of pixels of overlap between adjacent tiles.
        - origin (List[int], default=[0, 0]): List of coordinates [x, y] of the pixel from which the tiling starts (top-left corner of the first tile).

        Returns:
        None
        """
        self._image = image
        self._im_path = image.path
        self._w = int(self._image.width)
        self._h = int(self._image.height)
        self._origin = origin
        self._overlap = overlap
        self._nrow = grid[0]
        self._ncol = grid[1]
        self._limits = {}
        self._tiles = {}

    @property
    def grid(self) -> List[int]:
        """
        Get the grid size.

        Returns:
        List[int]: The number of rows and number of columns in the grid.
        """
        return [self._nrow, self._ncol]

    @property
    def limits(self) -> Dict[int, tuple]:
        """
        Get the tile limits.

        Returns:
        dict: A dictionary containing the index of each tile and its bounding box coordinates.
        """
        if not self._limits:
            raise ValueError(
                "Limits not available, compute them first with compute_limits_by_grid."
            )
        return self._limits

    def compute_limits_by_grid(self) -> None:
        """
        Compute the limits of each tile (i.e., xmin, ymin, xmax, ymax) given the number of rows and columns in the tile grid.

        Returns:
        None
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
                self._limits[tile_idx] = (xmin, ymin, xmax, ymax)

    def read_all_tiles(self) -> None:
        """
        Read all tiles and store them in the class instance.

        Returns:
        None
        """
        assert self._im_path is not None, "Invalid image path"
        for idx, limit in self._limits.items():
            self._tiles[idx] = self._image.extract_patch(limit)

    def read_tile(self, idx) -> np.ndarray:
        """
        Extract and return a tile given its index.

        Parameters:
        - idx (int): The index of the tile.

        Returns:
        np.ndarray: The extracted tile.
        """
        assert self._im_path is not None, "Invalid image path"
        return self._image.extract_patch(self._limits[idx])

    def remove_tiles(self, tile_idx=None) -> None:
        """
        Remove tiles from the class instance.

        Parameters:
        - tile_idx: The index of the tile to be removed. If None, remove all tiles.

        Returns:
        None
        """
        if tile_idx is None:
            self._tiles = {}
        else:
            self._tiles[tile_idx] = []

    def display_tiles(self) -> None:
        """
        Display all the stored tiles.

        Returns:
        None
        """
        for idx, tile in self._tiles.items():
            plt.subplot(self.grid[0], self.grid[1], idx + 1)
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

    t = tiles.read_tile(1)

    tiles.read_all_tiles()
    tiles.display_tiles()

    grid = tiles.grid
    limits = tiles.limits

    print(grid)

    print("Done")
