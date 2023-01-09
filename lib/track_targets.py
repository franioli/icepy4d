"""
MIT License

Copyright (c) 2022 Francesco Ioli

Permission is hereby granted, free of charge, to any person obtaining _A copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR _A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import gc

from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from typing import List, Union

from lib.base_classes.images import ImageDS
from lib.base_classes.targets import Targets

from lib.matching.templatematch import TemplateMatch, Stats, MatchResult
from lib.read_config import parse_yaml_cfg
from lib.utils.utils import AverageTimer
from lib.utils.inizialize_variables import Inizialization


class TrackTargets:
    def __init__(
        self,
        images: ImageDS,
        patch_centers: List[np.ndarray],
        patch_width: int = 512,
        method: str = "OC",
        template_width: int = 32,
        search_width: int = 128,
        base_epoch: int = 0,
        verbose: bool = True,
        debug_viz: bool = False,
    ) -> None:

        self.images = images
        self.patch_centers = patch_centers
        self.patch_width = patch_width
        self.method = method
        self.template_width = template_width
        self.search_width = search_width
        self.base_epoch = base_epoch

        self.verbose = verbose
        self.debug_viz = debug_viz

        self._iter = 0
        self._cur_target = 0

        # Initialize dictonary for storing results
        # Outer dictionary has one key for every target,
        # inner dictionary has one key for every epoch
        self.results = {}
        self.results[self._cur_target] = {}
        self.results[self._cur_target][self._iter] = {}

    def __iter__(self):
        return self

    def __next__(self):
        while self._iter < len(self.images):
            self._iter += 1
            return self._iter
        else:
            raise StopIteration

    def extract_pathes(self, epoch: int = None) -> None:
        # Coordinates of the center of the patch in full image
        if epoch is not None:
            iter = epoch
        else:
            iter = self._iter
        self._pc_int = np.round(self.patch_centers[self._cur_target]).astype(int)
        self._patch = [
            np.round(self._pc_int[0] - self.patch_width / 2).astype(int),
            np.round(self._pc_int[1] - self.patch_width / 2).astype(int),
            np.round(self._pc_int[0] + self.patch_width / 2).astype(int),
            np.round(self._pc_int[1] + self.patch_width / 2).astype(int),
        ]
        self._A = self.images.read_image(self.base_epoch).value[
            self._patch[1] : self._patch[3], self._patch[0] : self._patch[2]
        ]
        self._B = self.images.read_image(iter).value[
            self._patch[1] : self._patch[3], self._patch[0] : self._patch[2]
        ]

    def viz_template(self) -> None:
        """
        viz_template Visualize template on starting image with OpenCV
        """
        if not hasattr(self, "_A"):
            self.extract_pathes()
        patch_center = self._pc_int - self._patch[:2]
        template_coor = [
            (
                patch_center[0] - self.template_width,
                patch_center[1] - self.template_width,
            ),
            (
                patch_center[0] + self.template_width,
                patch_center[1] + self.template_width,
            ),
        ]
        win_name = "Template on image A"
        img = cv2.cvtColor(deepcopy(self._A), cv2.COLOR_BGR2RGB)
        cv2.circle(img, (patch_center[0], patch_center[1]), 0, (0, 255, 0), -1)
        cv2.rectangle(img, template_coor[0], template_coor[1], (0, 255, 0), 1)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        del img

    def track_single_epoch(self) -> None:
        self.extract_pathes()
        tm = TemplateMatch(
            A=cv2.cvtColor(self._A, cv2.COLOR_RGB2GRAY),
            B=cv2.cvtColor(self._B, cv2.COLOR_RGB2GRAY),
            xy=self.patch_centers[self._cur_target] - self._patch[:2],
            method=self.method,
            template_width=self.template_width,
            search_width=self.search_width,
        )
        r = tm.match()

        # Print result to std out
        if self.verbose:
            if r:
                print(
                    f"du: {r.du:.2f} dv: {r.dv:.2f} - Coor peak: {r.peakCorr:.2f} - SNR {r.snr:.2f}"
                )
            else:
                print("Target not found")

        # Create output image with the tracked target marked
        # TODO: pass output folder param to class
        if self.debug_viz:
            if r:
                self.viz_result("tmp/out", r)

        self.results[self._cur_target][self._iter] = r
        del self._A, self._B, tm, r
        gc.collect()

    def track(self) -> None:
        """
        track Perform tracking of the patch in all the following epoches by calling TemplateMatch
        """

        print(f"\tEpoch {self._iter}... ", end=" ")
        self.track_single_epoch()

        # Go to next epoch and call track recursively
        next(self)
        if self._iter < len(self.images):
            self.track()
        else:
            print("Tracking completed.")

    def track_all_targets(self):
        """
        track_all_targets run tracking for all the targets that are present in self.patch_centers
        """
        num_targets = self.patch_centers.shape[0]
        while self._cur_target < num_targets:
            print(f"Tracking target {self._cur_target}")
            self.results[self._cur_target] = {}
            self.track()
            self._iter = 0
            self._cur_target += 1

    def viz_result(self, out_dir: Union[str, Path], res: MatchResult) -> None:

        out_dir = Path(out_dir)
        x_est = self.patch_centers[self._cur_target, 0] + res.du
        y_est = self.patch_centers[self._cur_target, 1] + res.dv

        img = cv2.imread(str(self.images.get_image_path(self._iter)))
        cv2.drawMarker(
            img,
            (np.round(x_est).astype(int), np.round(y_est).astype(int)),
            (255, 0, 0),
            cv2.MARKER_CROSS,
            1,
        )
        cv2.imwrite(str(out_dir / self.images[self._iter]), img)

    def write_results_to_file(
        self,
        folder: Union[str, Path],
        targets_name: List[str],
        format: str = "csv",
        sep: str = ",",
    ) -> None:
        folder = Path(folder)
        self._cur_target = 0
        self._iter = 0
        num_targets = self.patch_centers.shape[0]

        self.images.reset_iterator()
        for ep, image in enumerate(self.images):
            f = open(folder / f"{image.stem}.{format}", "w")
            f.write(f"label{sep}x{sep}y\n")
            for i in range(num_targets):
                if self.results[i][ep]:
                    x_est = self.patch_centers[i, 0] + self.results[i][ep].du
                    y_est = self.patch_centers[i, 1] + self.results[i][ep].dv
                    f.write(f"{targets_name[i]}{sep}{x_est:.4f}{sep}{y_est:.4f}\n")
                else:
                    print(
                        f"Writing output error: target {targets_name[i]} not found on image {image.stem}"
                    )
            f.close()


if __name__ == "__main__":

    cfg_file = "config/config_2021.yaml"
    cfg = parse_yaml_cfg(cfg_file)
    init = Inizialization(cfg)
    init.inizialize_belpy()
    cams = init.cams
    images = init.images
    targets = init.targets

    # cam_id = 0
    cam_id = 1
    patch_width = 256
    targets_to_use = ["F2", "F12"]  #

    template_width = 16
    search_width = 64

    debug_viz = False
    debug = True

    t_est = {}
    diff = {}
    diff_noCC = {}

    targets_coord = np.zeros((len(targets_to_use), 2))
    for i, t in enumerate(targets_to_use):
        targets_coord[i] = targets[0].extract_image_coor_by_label([t], cam_id).squeeze()

    tracking = TrackTargets(
        images=images[cams[cam_id]],
        patch_centers=targets_coord,
        template_width=16,
        search_width=64,
        debug_viz=True,
    )
    # tracking.viz_template()
    # tracking.track()
    tracking.track_all_targets()
    tracking.write_results_to_file("tmp/out", targets_to_use)

    print("")
