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
import cv2
import gc

from copy import deepcopy
from pathlib import Path
from typing import List, Union

from icepy.base_classes.images import ImageDS
from icepy.base_classes.targets import Targets
from icepy.matching.templatematch import TemplateMatch, Stats, MatchResult
from icepy.utils.timer import AverageTimer


class TrackTargets:
    def __init__(
        self,
        images: ImageDS,
        patch_centers: List[np.ndarray],
        out_dir: str,
        patch_width: int = 512,
        method: str = "OC",
        template_width: int = 32,
        search_width: int = 128,
        base_epoch: int = 0,
        target_names: List[str] = None,
        verbose: bool = True,
        debug_viz: bool = False,
    ) -> None:

        self.images = images
        self.out_dir = Path(out_dir)
        self.patch_centers = patch_centers
        self.patch_width = patch_width
        self.method = method
        self.template_width = template_width
        self.search_width = search_width
        self.base_epoch = base_epoch

        self.target_names = target_names
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
        """
        extract_pathes Extract image patches for template matching.

        Args:
            epoch (int, optional): index of the image of ImageDS in which the target has to be search. If None, self._iter will be used. Defaults to None.
        """
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
        """
        track_single_epoch method to actual perform feature tracking on next image
        """
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
                self.viz_result(self.out_dir, r)

        self.results[self._cur_target][self._iter] = r
        del self._A, self._B, tm, r
        gc.collect()

    def track(self) -> None:
        """
        track Perform tracking of the patch in all the following epoches by calling TemplateMatch
        """

        print(f"\tEpoch {self._iter} - image: {self.images[self._iter]}... ", end=" ")

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
        timer = AverageTimer()

        num_targets = self.patch_centers.shape[0]
        while self._cur_target < num_targets:
            if self.target_names is not None:
                target_name = self.target_names[self._cur_target]
            else:
                target_name = self._cur_target
            print(f"Tracking target {target_name}")
            self.results[self._cur_target] = {}
            self.track()
            timer.update(f"target {target_name}")

            self._iter = 0
            self._cur_target += 1

        # Write targets to csv files
        self.write_results_to_file(self.out_dir, self.target_names)

    def viz_result(self, out_dir: Union[str, Path], res: MatchResult) -> None:
        """
        viz_result Visualize the tracked point on the image B patch and write image to disk

        Args:
            out_dir (Union[str, Path]): output directory where to save image
            res (MatchResult): MatchResult instance containing template matching result
        """
        out_dir = Path(out_dir) / "img"
        out_dir.mkdir(parents=True, exist_ok=True)
        x_est = res.pu + res.du
        y_est = res.pv + res.dv
        img = deepcopy(cv2.cvtColor(self._B, cv2.COLOR_RGB2BGR))
        cv2.drawMarker(
            img,
            (np.round(x_est).astype(int), np.round(y_est).astype(int)),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            2,
        )
        im_name = self.images.get_image_stem(self._iter)
        ext = self.images.get_image_path(self._iter).suffix
        if self.target_names is not None:
            target_name = self.target_names[self._cur_target]
        else:
            target_name = self._cur_target
        cv2.imwrite(
            str(out_dir / f"{target_name}_{im_name}{ext}"),
            img,
        )

    def write_results_to_file(
        self,
        folder: Union[str, Path],
        targets_name: List[str] = None,
        format: str = "csv",
        sep: str = ",",
        write_mode: str = "w",
    ) -> None:
        """
        write_results_to_file Write (full) image coordinates of the tracked target to csv file

        Args:
            folder (Union[str, Path]): output folder
            targets_name (List[str], optional): List containing the target names. If None, a numeric index will be used. Defaults to "None".
            format (str, optional): output file format. Defaults to "csv".
            sep (str, optional): field separator. Defaults to ",".
        """

        folder = Path(folder)
        self._cur_target = 0
        self._iter = 0
        num_targets = self.patch_centers.shape[0]
        if targets_name is None:
            targets_name = [x for x in range(num_targets)]

        for ep, image in enumerate(self.images):
            f = open(folder / f"{image.stem}.{format}", write_mode)
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
        print("Targets files saved correctely.")


if __name__ == "__main__":

    # TODO: implement tracking from and to a specific epoch

    # Parameters
    OUT_DIR = "tools/track_targets/block_3/results"

    TARGETS_DIR = "tools/track_targets/block_3/data/targets"
    TARGETS_IMAGES_FNAMES = ["IMG_2637.csv", "IMG_1112.csv"]
    TARGETS_WORLD_FNAME = "target_world.csv"
    IM_DIR = "tools/track_targets/block_3/data/images"

    cams = ["p1", "p2"]
    targets_to_track = ["F2", "F13"]

    patch_width = 512
    template_width = 16
    search_width = 64

    debug_viz = True
    verbose = True

    targets_image_paths = [Path(TARGETS_DIR) / fname for fname in TARGETS_IMAGES_FNAMES]
    targets = Targets(
        im_file_path=targets_image_paths,
        obj_file_path=Path(TARGETS_DIR) / TARGETS_WORLD_FNAME,
    )

    for cam_id, cam in enumerate(cams):
        print(f"Processing camera {cam}")

        images = ImageDS(Path(IM_DIR) / cam)

        # Define nx2 array with image coordinates of the targets to track
        # in the form of:
        # [x1, y1],
        # [x2, y2]...
        # You can create it manually or use Target class
        targets_coord = np.zeros((len(targets_to_track), 2))
        for i, target in enumerate(targets_to_track):
            targets_coord[i] = targets.get_image_coor_by_label([target], cam_id)[
                0
            ].squeeze()

        # Define TrackTargets object and run tracking
        tracking = TrackTargets(
            images=images,
            patch_centers=targets_coord,
            out_dir=OUT_DIR,
            target_names=targets_to_track,
            patch_width=patch_width,
            template_width=template_width,
            search_width=search_width,
            debug_viz=debug_viz,
            verbose=verbose,
        )
        # tracking.viz_template()
        tracking.track_all_targets()

        print("Done.")
