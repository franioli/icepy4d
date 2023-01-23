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

from pathlib import Path

from lib.base_classes.images import ImageDS, Image


if __name__ == "__main__":

    ROOT_DIR = "data/img"
    IM_DIR_LIST = ["p1", "p2"]
    IM_EXT = "jpg"

    ROOT_DIR = Path(ROOT_DIR)
    for dir in IM_DIR_LIST:
        folder = ROOT_DIR / dir
        images = ImageDS(folder, ext=IM_EXT)
        images.write_exif_to_csv(ROOT_DIR / f"image_list_{dir}.csv")

    print("Done")
