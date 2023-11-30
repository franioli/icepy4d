import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import List

import cv2
import numpy as np
from icepy4d.core import Image, Targets
from icepy4d.matching.templatematch import TemplateMatch
from icepy4d.utils import get_logger, setup_logger
from tqdm import tqdm

# Setup logger
setup_logger()
logger = get_logger(__name__)


class TrackTargets:
    # Define default config
    def_config = {
        "template_width": 32,
        "search_width": 128,
        "viz_tracked": False,
        "verbose": False,
        "snr_threshold": 7.0,
        "parallel": False,
        "num_workers": None,
    }
    valid_methods = ["OC"]  # , "NCC"]

    def __init__(
        self,
        master: Path,
        images: List[Image],
        targets: np.ndarray,
        method: str = "OC",
        out_dir: str = "results",
        target_names: List[str] = None,
        **config,
    ) -> None:
        if not isinstance(images, list):
            raise TypeError("images must be a list of Image objects")

        if not isinstance(master, Path):
            raise TypeError(
                "master must be a Path object with the path to the master image"
            )

        if not isinstance(targets, np.ndarray) or targets.shape[1] != 2:
            raise TypeError(
                "targets must be a numpy vector of shape (n, 2) containing the image coordinates of the targets to track"
            )

        if method not in self.valid_methods:
            raise ValueError(
                f"Method {method} currentely not supported. Use {self.valid_methods}"
            )

        # Update default config with user config
        self.cfg = {**self.def_config, **config}

        # Store input parameters
        self.images = images
        self.targets = targets
        self.target_names = target_names
        self.method = method
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Load master image
        self._master = cv2.imread(str(master), cv2.IMREAD_GRAYSCALE)
        self._slave = None

        # Initialize dictonary for storing results
        self.results = {}

    def track_image(self, slave: Image) -> dict:
        """
        track_single_epoch method to actual perform feature tracking on next image
        """
        slave_name = slave.name
        target_names = self.target_names
        template_width = self.cfg["template_width"]
        search_width = self.cfg["search_width"]
        snr_threshold = self.cfg["snr_threshold"]
        viz_tracked = self.cfg["viz_tracked"]
        verbose = self.cfg["verbose"]

        slave_img = cv2.imread(str(str(slave.path)), cv2.IMREAD_GRAYSCALE)

        tm = TemplateMatch(
            A=self._master,
            B=slave_img,
            xy=self.targets,
            method=self.method,
            template_width=template_width,
            search_width=search_width,
            single_points=True,
        )
        r = tm.match()

        if not r:
            logger.info(f"No target found in image {slave_name}")
            return None

        # Save result to dict
        du = np.diag(r.du)
        dv = np.diag(r.dv)
        x_est = self.targets[:, 0] + du
        y_est = self.targets[:, 1] + dv
        snr = np.diag(r.snr)
        peak_corr = np.diag(r.peakCorr)

        result = {
            "image": slave_name,
            "targets_names": target_names,
            "targets_coord": self.targets,
            "pu": np.diag(r.pu),
            "pv": np.diag(r.pv),
            "du": du,
            "dv": dv,
            "x_est": x_est,
            "y_est": y_est,
            "snr": snr,
            "peak_corr": peak_corr,
            "meanAbsCorr": np.diag(r.meanAbsCorr),
        }

        # Print result to std out
        if verbose:
            msg = ""
            for n, u, v, s, p in zip(target_names, du, dv, snr, peak_corr):
                if s > snr_threshold:
                    msg += f"{slave_name}\t\t{n}\t{u:.2f}\t\t{v:.2f}\t\t{s:.2f}\t\t{p:.2f}\n"
                else:
                    msg += f"{slave_name}\t\t{n}\tRejected\n"
            print(msg)

        # Write result to file
        # TODO: move to a separate method
        fname = self.out_dir / f"{slave_name}.csv"
        with open(fname, "w") as f:
            f.write("label,x,y\n")
            for name, x, y, s in zip(target_names, x_est, y_est, snr):
                if s > snr_threshold:
                    f.write(f"{name},{x:.3f},{y:.3f}\n")

        # Visualize result
        # TODO: move to a separate method
        if viz_tracked:
            img_dir = self.out_dir / "img"
            img_dir.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(slave.path))
            for x, y, s in zip(x_est, y_est, snr):
                if s > snr_threshold:
                    cv2.drawMarker(
                        img,
                        (np.round(x).astype(int), np.round(y).astype(int)),
                        (0, 255, 0),
                        cv2.MARKER_CROSS,
                        2,
                    )
                elif not np.isnan(x):
                    cv2.drawMarker(
                        img,
                        (np.round(x).astype(int), np.round(y).astype(int)),
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        2,
                    )
            cv2.imwrite(
                str(img_dir / f"{slave_name}.jpg"),
                img,
            )

        return result

    def track(self) -> None:
        """
        Perform tracking of the patch in all the following epochs by calling TemplateMatch
        """
        if self.cfg["verbose"]:
            print("Image\t\t\t\t\ttarget\tdu\t\tdv\t\tSNR\t\tPeak Corr")

        if self.cfg["parallel"]:
            logger.info("Running in parallel mode")
            # # Use multiprocessing Pool to parallelize the execution of the tracking
            with Pool(processes=self.cfg["num_workers"]) as pool:
                self.results = pool.map(self.track_image, self.images)
        else:
            # Run non-parallel version for comparison
            for slave in tqdm(self.images):
                self.results[slave.name] = self.track_image(slave)

        logger.info("Tracking completed.")


if __name__ == "__main__":
    # Define input parameters
    cams = ["p1", "p2"]
    IM_DIRS = ["data/img/p1", "data/img/p2"]
    MASTER_IMAGES = [
        "p1_20230725_135953_IMG_1149.JPG",
        "p2_20230725_140026_IMG_0887.JPG",
    ]
    TARGETS_DIR = "data/targets"
    TARGETS_WORLD_FNAME = "targets_world.csv"
    TARGETS_IMAGES_FNAMES = [
        "p1_20230725_135953_IMG_1149.csv",
        "p2_20230725_140026_IMG_0887.csv",
    ]
    OUT_DIR = "tracking_res"

    # Define targets to track
    targets_to_track = ["F2", "F10", "T3"]

    # Template matching parameters
    template_width = 32
    search_width = 128

    for id, cam in enumerate(cams):
        # Build Targets object
        target_dir = Path(TARGETS_DIR)
        targets = Targets(
            im_file_path=[target_dir / f for f in TARGETS_IMAGES_FNAMES],
            obj_file_path=target_dir / TARGETS_WORLD_FNAME,
        )

        # Build list of Image objects
        img_dir = Path(IM_DIRS[id])
        images = [Image(f) for f in sorted(img_dir.glob("*"))]
        master = img_dir / MASTER_IMAGES[id]
        out_dir = Path(OUT_DIR) / cam
        if out_dir.exists():
            shutil.rmtree(out_dir)

        # Define nx2 array with image coordinates of the targets to track
        # in the form of:
        # [x1, y1],
        # [x2, y2]...
        # You can create it manually or use Target class
        targets_coord, _ = targets.get_image_coor_by_label(targets_to_track, id)

        # Define TrackTarget object and run tracking
        logger.info(f"Tracking targets {targets_to_track} in camera {cam}")
        tracking = TrackTarget(
            master=master,
            images=images,
            targets=targets_coord.squeeze(),
            out_dir=out_dir,
            target_names=targets_to_track,
            template_width=template_width,
            search_width=search_width,
            verbose=True,
            viz_tracked=True,
            parallel=True,
        )
        tracking.track()

    print("Done.")
