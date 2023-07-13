"""SuperGlue matcher implementation
The network was proposed in 'SuperGlue: Learning Feature Matching with Graph Neural Networks' and is implemented by wrapping over author's source-code.
Note: the pretrained model only supports SuperPoint detections currently.
References:
- http://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
- https://github.com/magicleap/SuperGluePretrainedNetwork
"""
import importlib
import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict

from icepy4d.thirdparty.SuperGlue.models.matching import Matching
from icepy4d.thirdparty.SuperGlue.models.utils import make_matching_plot
from icepy4d.utils import AverageTimer, timeit

matplotlib.use("TkAgg")


# TODO: use KORNIA for image tiling

# SuperPoint and SuperGlue default parameters
NMS_RADIUS = 3
SUPERGLUE_DESC_DIM = 256
SINKHORN_ITERATIONS = 10

# Set logging level
LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)


class TileSelection(Enum):
    """Enumeration for tile selection methods."""

    NONE = 0
    EXHAUSTIVE = 1
    GRID = 2
    PRESELECTION = 3


class GeometricVerification(Enum):
    """Enumeration for geometric verification methods."""

    NONE = 1
    PYDEGENSAC = 2
    MAGSAC = 3


class Quality(Enum):
    """Enumeration for matching quality."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    HIGHEST = 4


@dataclass
class feature:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray


class Tiler:
    """
    Class for dividing an image into tiles.
    """

    def __init__(
        self,
        grid: List[int] = [1, 1],
        overlap: int = 0,
        origin: List[int] = [0, 0],
        max_length: int = 2000,
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
        self._origin = origin
        self._overlap = overlap
        self._nrow = grid[0]
        self._ncol = grid[1]
        self._limits = None
        self._tiles = None

    @property
    def grid(self) -> List[int]:
        """
        Get the grid size.

        Returns:
        List[int]: The number of rows and number of columns in the grid.
        """
        return [self._nrow, self._ncol]

    @property
    def origin(self) -> List[int]:
        """
        Get the origin of the tiling.

        Returns:
        List[int]: The coordinates [x, y] of the pixel from which the tiling starts (top-left corner of the first tile).
        """
        return self._origin

    @property
    def overlap(self) -> int:
        """
        Get the overlap size.

        Returns:
        int: The number of pixels of overlap between adjacent tiles.
        """
        return self._overlap

    @property
    def limits(self) -> Dict[int, tuple]:
        """
        Get the tile limits.

        Returns:
        dict: A dictionary containing the index of each tile and its bounding box coordinates.
        """
        return self._limits

    def compute_grid_size(self, max_length: int) -> None:
        """
        Compute the best number of rows and columns for the grid based on the maximum length of each tile.

        Parameters:
        - max_length (int): The maximum length of each tile.

        Returns:
        None

        NOTE: NOT WORKING. NEEDS TO BE TESTED.
        """
        self._nrow = int(np.ceil(self._h / (max_length - self._overlap)))
        self._ncol = int(np.ceil(self._w / (max_length - self._overlap)))

    def compute_limits_by_grid(self, image: np.ndarray) -> List[int]:
        """
        Compute the limits of each tile (i.e., xmin, ymin, xmax, ymax) given the number of rows and columns in the tile grid.

        Returns:
        List[int]: A list containing the bounding box coordinates of each tile as: [xmin, ymin, xmax, ymax]
        List[int]: The coordinates [x, y] of the pixel from which the tiling starts (top-left corner of the first tile).
        """

        self._image = image
        self._w = image.shape[1]
        self._h = image.shape[0]

        DX = round((self._w - self._origin[0]) / self._ncol / 10) * 10
        DY = round((self._h - self._origin[1]) / self._nrow / 10) * 10

        self._limits = {}
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

        return self._limits, self._origin

    def extract_patch(self, image: np.ndarray, limits: List[int]) -> np.ndarray:
        """Extract image patch
        Parameters
        __________
        - limits (List[int]): List containing the bounding box coordinates as: [xmin, ymin, xmax, ymax]
        __________
        Return: patch (np.ndarray)
        """
        patch = image[
            limits[1] : limits[3],
            limits[0] : limits[2],
        ]
        return patch

    def read_all_tiles(self) -> None:
        """
        Read all tiles and store them in the class instance.

        Returns:
        None
        """
        self._tiles = {}
        for idx, limit in self._limits.items():
            self._tiles[idx] = self.extract_patch(self._image, limit)

    def read_tile(self, idx) -> np.ndarray:
        """
        Extract and return a tile given its index.

        Parameters:
        - idx (int): The index of the tile.

        Returns:
        np.ndarray: The extracted tile.
        """
        if self._tiles is None:
            self._tiles = {}
        return self.extract_patch(self._image, self._limits[idx])

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


def build_superglue_config(opt: dict) -> dict:
    def_opt = {
        "weights": "outdoor",
        "keypoint_threshold": 0.001,
        "max_keypoints": -1,
        "match_threshold": 0.3,
        "force_cpu": False,
        "nms_radius": NMS_RADIUS,
        "sinkhorn_iterations": SINKHORN_ITERATIONS,
    }
    opt = {**def_opt, **opt}
    required_keys = [
        "weights",
        "keypoint_threshold",
        "max_keypoints",
        "match_threshold",
        "force_cpu",
    ]
    check_dict_keys(opt, required_keys)

    return {
        "superpoint": {
            "nms_radius": opt["nms_radius"],
            "keypoint_threshold": opt["keypoint_threshold"],
            "max_keypoints": opt["max_keypoints"],
        },
        "superglue": {
            "weights": opt["weights"],
            "sinkhorn_iterations": opt["sinkhorn_iterations"],
            "match_threshold": opt["match_threshold"],
        },
        "force_cpu": opt["force_cpu"],
    }


def geometric_verification(
    mkpts0: np.ndarray = None,
    mkpts1: np.ndarray = None,
    method: GeometricVerification = GeometricVerification.PYDEGENSAC,
    threshold: float = 1,
    confidence: float = 0.9999,
    max_iters: int = 10000,
    laf_consistensy_coef: float = -1.0,
    error_type: str = "sampson",
    symmetric_error_check: bool = True,
    enable_degeneracy_check: bool = True,
) -> dict:
    """
    Computes the fundamental matrix and inliers between the two images using geometric verification.

    Args:
        method (str): The method used for geometric verification. Can be one of ['pydegensac', 'opencv'].
        threshold (float): Pixel error threshold for considering a correspondence an inlier.
        confidence (float): The required confidence level in the results.
        max_iters (int): The maximum number of iterations for estimating the fundamental matrix.
        laf_consistensy_coef (float): The weight given to Local Affine Frame (LAF) consistency term for pydegensac.
        error_type (str): The error function used for computing the residuals in the RANSAC loop.
        symmetric_error_check (bool): If True, performs an additional check on the residuals in the opposite direction.
        enable_degeneracy_check (bool): If True, enables the check for degeneracy using SVD.

    Returns:
        np.ndarray: A Boolean array that masks the correspondences that were identified as inliers.

    TODO: allow input parameters for both pydegensac and MAGSAC.

    """

    assert isinstance(
        method, GeometricVerification
    ), "Invalid method. It must be a GeometricVerification enum in GeometricVerification.PYDEGENSAC or GeometricVerification.MAGSAC."

    if method == GeometricVerification.PYDEGENSAC:
        try:
            pydegensac = importlib.import_module("pydegensac")
            fallback = False
        except:
            logging.error(
                "Pydegensac not available. Using MAGSAC++ (OpenCV) for geometric verification."
            )
            fallback = True

    if method == GeometricVerification.PYDEGENSAC and not fallback:
        try:
            F, inlMask = pydegensac.findFundamentalMatrix(
                mkpts0,
                mkpts1,
                px_th=threshold,
                conf=confidence,
                max_iters=max_iters,
                laf_consistensy_coef=laf_consistensy_coef,
                error_type=error_type,
                symmetric_error_check=symmetric_error_check,
                enable_degeneracy_check=enable_degeneracy_check,
            )
            logging.info(
                f"Pydegensac found {inlMask.sum()} inliers ({inlMask.sum()*100/len(mkpts0):.2f}%)"
            )
        except Exception as err:
            # Fall back to MAGSAC++ if pydegensac fails
            logging.error(
                f"{err}. Unable to perform geometric verification with Pydegensac. Trying using MAGSAC++ (OpenCV) instead."
            )
            fallback = True

    if method == GeometricVerification.MAGSAC or fallback:
        try:
            F, inliers = cv2.findFundamentalMat(
                mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000
            )
            inlMask = (inliers > 0).squeeze()
            logging.info(
                f"MAGSAC++ found {inlMask.sum()} inliers ({inlMask.sum()*100/len(mkpts0):.2f}%)"
            )
        except Exception as err:
            logging.error(
                f"{err}. Unable to perform geometric verification with MAGSAC++."
            )
            inlMask = np.ones(len(mkpts0), dtype=bool)

    return F, inlMask


def check_dict_keys(dict: dict, keys: List[str]):
    missing_keys = [key for key in keys if key not in dict]
    if missing_keys:
        raise KeyError(
            f"Missing required keys: {', '.join(missing_keys)} Matcher option dictionary"
        )


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.0).float()[None, None].to(device)


class ImageMatcherABC(ABC):
    def __init__(self, opt: dict = {}) -> None:
        self._opt = edict(opt)

    @abstractmethod
    def match(self):
        pass

    @abstractmethod
    def _match_images(self):
        pass

    @abstractmethod
    def _match_tiles(self):
        pass


class ImageMatcherBase(ImageMatcherABC):
    def __init__(self, opt: dict = {}) -> None:
        """Base class for matchers"""
        if not isinstance(opt, dict):
            raise TypeError("opt must be a dictionary")
        self._opt = edict(opt)
        self._device = (
            "cuda" if torch.cuda.is_available() and not opt.get("force_cpu") else "cpu"
        )
        logging.info(f"Running inference on device {self._device}")

    @property
    def device(self):
        return self._device

    def match(self):
        """
        Matches keypoints between two images. This method must be implemented by the child class.
        """
        raise NotImplementedError("Subclasses must implement match() method.")

    def _match_images(self):
        raise NotImplementedError(
            "Subclasses must implement _match_full_images() method."
        )

    def _match_tiles(self):
        raise NotImplementedError("Subclasses must implement _match_tiles() method.")


class SuperGlueMatcher(ImageMatcherBase):
    def __init__(self, opt: dict) -> None:
        """Initializes a SuperGlueMatcher object with the given options dictionary.

        The options dictionary should contain the following keys:

        - 'superglue': path to the SuperGlue model weights file
        - 'keypoint_threshold': threshold for the SuperPoint keypoint detector
        - 'max_keypoints': maximum number of keypoints to extract with the SuperPoint detector
        - 'match_threshold': threshold for the SuperGlue feature matcher
        - 'force_cpu': whether to force using the CPU for inference

        Args:
            opt (dict): a dictionary of options for configuring the SuperGlueMatcher object

        Raises:
            KeyError: if one or more required options are missing from the options dictionary
            FileNotFoundError: if the specified SuperGlue model weights file cannot be found

        """

        opt = build_superglue_config(opt)
        super().__init__(opt)

        # Inizialize additional variable members for storing matched keypoints descriptors and scores
        self._mkpts0 = None  # matched keypoints on image 0
        self._mkpts1 = None  # matched keypoints on image 1
        self._descriptors0 = None  # descriptors of mkpts on image 0
        self._descriptors1 = None  # descriptors of mkpts on image 1
        self._scores0 = None  # scores of mkpts on image 0
        self._scores1 = None  # scores of mkpts on image 1
        self._mconf = None  # match confidence (i.e., scores0 of the valid matches)

    def reset(self):
        """Reset the matcher by clearing the features and matches"""
        self._mkpts0 = None  # matched keypoints on image 0
        self._mkpts1 = None  # matched keypoints on image 1
        self._descriptors0 = None  # descriptors of mkpts on image 0
        self._descriptors1 = None  # descriptors of mkpts on image 1
        self._scores0 = None  # scores of mkpts on image 0
        self._scores1 = None  # scores of mkpts on image 1
        self._mconf = None  # match confidence (i.e., scores0 of the valid matches)

    @timeit
    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        quality: Quality = Quality.HIGH,
        tile_selection: TileSelection = TileSelection.PRESELECTION,
        **kwargs,
    ):
        self.timer = AverageTimer()

        # Get kwargs
        do_viz_matches = kwargs.get("do_viz_matches", False)
        save_dir = kwargs.get("save_dir", ".")
        gv_method = kwargs.get(
            "geometric_verification", GeometricVerification.PYDEGENSAC
        )
        threshold = kwargs.get("threshold", 1)
        confidence = kwargs.get("confidence", 0.9999)

        # Inizialize the Matching object with given configuration
        self.matcher = Matching(self._opt).eval().to(self._device)

        # Resize images if needed
        image0_, image1_ = self._resize_images(quality, image0, image1)

        # Perform matching (on tiles or full images)
        if tile_selection == TileSelection.NONE:
            logging.info("Matching full images...")
            features0, features1, matches0 = self._match_images(image0_, image1_)

        else:
            logging.info("Matching by tiles...")
            features0, features1, matches0 = self._match_tiles(
                image0_, image1_, tile_selection, **kwargs
            )

        # Retrieve original image coordinates if matching was performed on up/down-sampled images
        features0, features1 = self._resize_features(quality, features0, features1)

        # Store features as class members
        self._store_features(features0, features1, matches0)
        self.timer.update("matching")
        logging.info("Matching done!")

        # Perform geometric verification
        logging.info("Performing geometric verification...")
        if gv_method is not GeometricVerification.NONE:
            F, inlMask = geometric_verification(
                self._mkpts0,
                self._mkpts1,
                method=gv_method,
                confidence=confidence,
                threshold=threshold,
            )
            self._F = F
            self._filter_matches_by_mask(inlMask)
            logging.info("Geometric verification done.")
            self.timer.update("geometric_verification")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if do_viz_matches is True:
            self.viz_matches(image0, image1, save_dir / f"matches.png")

        self.timer.print("Matching")

    def _resize_images(
        self, quality: Quality, image0: np.ndarray, image1: np.ndarray
    ) -> Tuple[np.ndarray]:
        if quality == Quality.HIGHEST:
            image0_ = cv2.pyrUp(image0)
            image1_ = cv2.pyrUp(image1)
        elif quality == Quality.HIGH:
            image0_ = image0
            image1_ = image1
        elif quality == Quality.MEDIUM:
            image0_ = cv2.pyrDown(image0)
            image1_ = cv2.pyrDown(image1)
        elif quality == Quality.LOW:
            image0_ = cv2.pyrDown(cv2.pyrDown(image0))
            image1_ = cv2.pyrDown(cv2.pyrDown(image1))
        return image0_, image1_

    def _resize_features(
        self, quality: Quality, features0: feature, features1: feature
    ) -> Tuple[feature]:
        if quality == Quality.HIGHEST:
            features0.keypoints /= 2
            features1.keypoints /= 2
        elif quality == Quality.HIGH:
            pass
        elif quality == Quality.MEDIUM:
            features0.keypoints *= 2
            features1.keypoints *= 2
        elif quality == Quality.LOW:
            features0.keypoints *= 4
            features1.keypoints *= 4

        return features0, features1

    def _tile_selection(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        t0_lims: dict,
        t1_lims: dict,
        method: TileSelection = TileSelection.PRESELECTION,
    ):
        def points_in_rect(points: np.ndarray, rect: np.ndarray) -> np.ndarray:
            logic = np.all(points > rect[:2], axis=1) & np.all(
                points < rect[2:], axis=1
            )
            return logic

        # default parameters
        min_matches_per_tile = 10

        # Select tile pairs to match
        if method == TileSelection.EXHAUSTIVE:
            logging.info("Matching tiles exaustively")
            tile_pairs = sorted(product(t0_lims.keys(), t1_lims.keys()))
        elif method == TileSelection.GRID:
            logging.info("Matching tiles by regular grid")
            tile_pairs = sorted(zip(t0_lims.keys(), t1_lims.keys()))
        elif method == TileSelection.PRESELECTION:
            logging.info("Matching tiles by preselection tile selection")
            if image0.shape[0] > 8000:
                n_down = 4
            if image0.shape[0] > 4000:
                n_down = 3
            elif image0.shape[0] > 2000:
                n_down = 2
            else:
                n_down = 1

            i0 = deepcopy(image0)
            i1 = deepcopy(image1)
            for _ in range(n_down):
                i0 = cv2.pyrDown(i0)
                i1 = cv2.pyrDown(i1)
            f0, f1, mtc = self._match_images(i0, i1)
            vld = mtc > -1
            kp0 = f0.keypoints[vld]
            kp1 = f1.keypoints[mtc[vld]]
            for _ in range(n_down):
                kp0 *= 2
                kp1 *= 2

            tile_pairs = []
            all_pairs = sorted(product(t0_lims.keys(), t1_lims.keys()))
            for tidx0, tidx1 in all_pairs:
                lim0 = t0_lims[tidx0]
                lim1 = t1_lims[tidx1]
                ret0 = points_in_rect(kp0, lim0)
                ret1 = points_in_rect(kp1, lim1)
                ret = ret0 & ret1
                if sum(ret) > min_matches_per_tile:
                    tile_pairs.append((tidx0, tidx1))
            self.timer.update("preselection")

        return tile_pairs

    def _match_tiles(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        tile_selection: TileSelection = TileSelection.PRESELECTION,
        **kwargs,
    ):
        assert isinstance(image0, np.ndarray), "image0 must be a NumPy array"
        assert isinstance(image1, np.ndarray), "image1 must be a NumPy array"

        # Get kwargs
        grid = kwargs.get("grid", [1, 1])
        overlap = kwargs.get("overlap", 0)
        origin = kwargs.get("origin", [0, 0])
        do_viz_tiles = kwargs.get("do_viz_tiles", False)

        # Convert images to grayscale if needed
        if len(image0.shape) > 2:
            image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
        if len(image1.shape) > 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

        # Compute tiles limits and origin
        self._tiler = Tiler(grid=grid, overlap=overlap, origin=origin)
        t0_lims, t0_origin = self._tiler.compute_limits_by_grid(image0)
        t1_lims, t1_origin = self._tiler.compute_limits_by_grid(image1)

        # Select tile pairs to match
        tile_pairs = self._tile_selection(
            image0, image1, t0_lims, t1_lims, tile_selection
        )

        # Initialize empty array for storing matched keypoints, descriptors and scores
        mkpts0_full = np.array([], dtype=np.float32).reshape(0, 2)
        mkpts1_full = np.array([], dtype=np.float32).reshape(0, 2)
        descriptors0_full = np.array([], dtype=np.float32).reshape(256, 0)
        descriptors1_full = np.array([], dtype=np.float32).reshape(256, 0)
        scores0_full = np.array([], dtype=np.float32)
        scores1_full = np.array([], dtype=np.float32)
        conf_full = np.array([], dtype=np.float32)

        # Match each tile pair
        for tidx0, tidx1 in tile_pairs:
            logging.info(f" - Matching tile pair ({tidx0}, {tidx1})")

            lim0 = t0_lims[tidx0]
            lim1 = t1_lims[tidx1]
            tile0 = self._tiler.extract_patch(image0, lim0)
            tile1 = self._tiler.extract_patch(image1, lim1)

            # Run SuperGlue on a pair of tiles
            tensor0 = frame2tensor(tile0, self._device)
            tensor1 = frame2tensor(tile1, self._device)
            with torch.inference_mode():
                pred_tensor = self.matcher({"image0": tensor0, "image1": tensor1})
            pred = {k: v[0].cpu().numpy() for k, v in pred_tensor.items()}

            # Get matches
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            descriptors0, descriptors1 = (
                pred["descriptors0"],
                pred["descriptors1"],
            )
            scores0, scores1 = pred["scores0"], pred["scores1"]
            matches0, matches1 = (
                pred["matches0"],
                pred["matches1"],
            )

            conf = pred["matching_scores0"]
            valid = matches0 > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches0[valid]]
            descriptors0 = descriptors0[:, valid]
            descriptors1 = descriptors1[:, matches0[valid]]
            scores0 = scores0[valid]
            scores1 = scores1[matches0[valid]]
            conf = conf[valid]

            mkpts0_full = np.vstack(
                (mkpts0_full, mkpts0 + np.array(lim0[0:2]).astype("float32"))
            )
            mkpts1_full = np.vstack(
                (mkpts1_full, mkpts1 + np.array(lim1[0:2]).astype("float32"))
            )
            descriptors0_full = np.hstack((descriptors0_full, descriptors0))
            descriptors1_full = np.hstack((descriptors1_full, descriptors1))
            scores0_full = np.concatenate((scores0_full, scores0))
            scores1_full = np.concatenate((scores1_full, scores1))
            conf_full = np.concatenate((conf_full, conf))

            save_dir = kwargs.get("save_dir", ".")
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            if do_viz_tiles is True:
                self._viz_matches_mpl(
                    tile0,
                    tile1,
                    mkpts0,
                    mkpts1,
                    save_dir / f"matches_tile_{tidx0}-{tidx1}.png",
                )

        logging.info("Restoring full image coordinates of matches...")

        # Restore original image coordinates (not cropped)
        mkpts0_full = mkpts0_full + np.array(t0_origin).astype("float32")
        mkpts1_full = mkpts1_full + np.array(t1_origin).astype("float32")

        # Select uniue features on image 0
        mkpts0_full, unique_idx = np.unique(mkpts0_full, axis=0, return_index=True)
        descriptors0_full = descriptors0_full[:, unique_idx]
        scores0_full = scores0_full[unique_idx]
        mkpts1_full = mkpts1_full[unique_idx]
        descriptors1_full = descriptors1_full[:, unique_idx]
        scores1_full = scores1_full[unique_idx]

        # Create features
        features0 = feature(
            keypoints=mkpts0_full, descriptors=descriptors0_full, scores=scores0_full
        )
        features1 = feature(
            keypoints=mkpts1_full, descriptors=descriptors1_full, scores=scores1_full
        )

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0_full.shape[0])

        logging.info("Matching by tile completed.")

        return features0, features1, matches0

    def _match_images(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
    ) -> Tuple[feature, feature, np.ndarray]:
        """Matches keypoints and descriptors in two given images (no matter if they are tiles or full-res images) using the SuperGlue algorithm.

        This method takes in two images as Numpy arrays, and returns the matches between keypoints
        and descriptors in those images using the SuperGlue algorithm.

        Args:
            image0 (np.ndarray): the first image to match, as Numpy array
            image1 (np.ndarray): the second image to match, as Numpy array

        Returns:
            Tuple[feature, feature, np.ndarray]: a tuple containing the features of the first image, the features of the second image, and the matches between them
        """

        if len(image0.shape) > 2:
            image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
        if len(image1.shape) > 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

        tensor0 = frame2tensor(image0, self._device)
        tensor1 = frame2tensor(image1, self._device)

        with torch.inference_mode():
            pred_tensor = self.matcher({"image0": tensor0, "image1": tensor1})
        pred = {k: v[0].cpu().numpy() for k, v in pred_tensor.items()}

        features0 = feature(
            keypoints=pred["keypoints0"],
            descriptors=pred["descriptors0"],
            scores=pred["scores0"],
        )
        features1 = feature(
            keypoints=pred["keypoints1"],
            descriptors=pred["descriptors1"],
            scores=pred["scores1"],
        )
        matches0 = pred["matches0"]

        return features0, features1, matches0

    def _store_features(
        self,
        features0: feature,
        features1: feature,
        matches0: np.ndarray,
        force_overwrite: bool = True,
    ) -> None:
        """Stores keypoints, descriptors and scores of the matches in the object's members."""

        assert isinstance(features0, feature), "features0 must be a feature object"
        assert isinstance(features1, feature), "features1 must be a feature object"
        assert hasattr(features0, "keypoints"), "No keypoints found in features0"
        assert hasattr(features1, "keypoints"), "No keypoints found in features1"

        if self._mkpts0 is not None and self._mkpts1 is not None:
            if force_overwrite is False:
                logging.warning(
                    "Matches already stored. Not overwriting them. Use force_overwrite=True to force overwrite them."
                )
                return
            else:
                logging.warning("Matches already stored. Overwrite them")

        valid = matches0 > -1
        self._valid = valid
        idx1 = matches0[valid]
        self._mkpts0 = features0.keypoints[valid]
        self._mkpts1 = features1.keypoints[idx1]
        self._descriptors0 = features0.descriptors[:, valid]
        self._descriptors1 = features1.descriptors[:, idx1]
        self._scores0 = features0.scores[valid]
        self._scores1 = features1.scores[idx1]
        self._mconf = features0.scores[valid]

    def _filter_matches_by_mask(self, inlMask: np.ndarray) -> None:
        self._mkpts0 = self._mkpts0[inlMask, :]
        self._mkpts1 = self._mkpts1[inlMask, :]
        if self._descriptors0 is not None:
            self._descriptors0 = self._descriptors0[:, inlMask]
        if self._descriptors1 is not None:
            self._descriptors1 = self._descriptors1[:, inlMask]
        if self._scores0 is not None:
            self._scores0 = self._scores0[inlMask]
        if self._scores1 is not None:
            self._scores1 = self._scores1[inlMask]
        if self._mconf is not None:
            self._mconf = self._mconf[inlMask]

    def _viz_matches_mpl(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        save_path: str = None,
        hide_fig: bool = True,
        **kwargs,
    ) -> None:
        colors = kwargs.get("c", kwargs.get("color", ["r", "r"]))
        if isinstance(colors, str):
            colors = [colors, colors]
        s = kwargs.get("s", 5)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(cv2.cvtColor(image0, cv2.COLOR_BAYER_BG2BGR))
        ax[0].scatter(kpts0[:, 0], kpts0[:, 1], s=s, c=colors[0])
        ax[1].imshow(cv2.cvtColor(image1, cv2.COLOR_BAYER_BG2BGR))
        ax[1].scatter(kpts1[:, 0], kpts1[:, 1], s=s, c=colors[1])
        if save_path is not None:
            fig.savefig(save_path)
        if hide_fig is False:
            plt.show()
        else:
            plt.close(fig)

    def viz_matches(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        path: str,
        fast_viz: bool = False,
        show_keypoints: bool = False,
        opencv_display: bool = False,
    ) -> None:
        """
        Visualize the matching result between two images.

        Args:
        - path (str): The output path to save the visualization.
        - fast_viz (bool): Whether to use preselection visualization method.
        - show_keypoints (bool): Whether to show the detected keypoints.
        - opencv_display (bool): Whether to use OpenCV for displaying the image.

        Returns:
        - None

        TODO: replace make_matching_plot with native a function implemented in icepy4D
        """

        assert self._mkpts0 is not None, "Matches not available."
        # image0 = np.uint8(tensor0.cpu().numpy() * 255),

        color = cm.jet(self._mconf)
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(
                len(self._mkpts0),
                len(self._mkpts1),
            ),
            "Matches: {}".format(len(self._mkpts0)),
        ]

        # Display extra parameter info.
        k_thresh = self._opt["superpoint"]["keypoint_threshold"]
        m_thresh = self._opt["superglue"]["match_threshold"]
        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
        ]

        make_matching_plot(
            image0,
            image1,
            self._mkpts0,
            self._mkpts1,
            self._mkpts0,
            self._mkpts1,
            color,
            text,
            path=path,
            show_keypoints=show_keypoints,
            fast_viz=fast_viz,
            opencv_display=opencv_display,
            opencv_title="Matches",
            small_text=small_text,
        )


if __name__ == "__main__":
    import os

    assset_path = Path("assets")

    # im_path0 = assset_path / "img/cam1/IMG_2637.jpg"
    # im_path1 = assset_path / "img/cam2/IMG_1112.jpg"

    folders = [Path("data/img/p1"), Path("data/img/p2")]
    imlists = [sorted(f.glob("*.jpg")) for f in folders]

    img_idx = 10
    im_path0 = imlists[0][10]
    im_path1 = imlists[1][10]
    img0 = cv2.imread(str(im_path0))
    img1 = cv2.imread(str(im_path1))
    outdir = "mmm"
    outdir = Path(outdir)
    if outdir.exists():
        os.system(f"rm -rf {outdir}")

    suerglue_cfg = {
        "weights": "outdoor",
        "keypoint_threshold": 0.001,
        "max_keypoints": 4096,
        "match_threshold": 0.1,
        "force_cpu": False,
    }
    matcher = SuperGlueMatcher(suerglue_cfg)
    # mkpts = matcher.match(img0, img1)

    grid = [4, 3]
    overlap = 200
    origin = [0, 0]

    tile_selection = TileSelection.PRESELECTION
    matcher.match(
        img0,
        img1,
        quality=Quality.HIGHEST,
        tile_selection=tile_selection,
        grid=[5, 4],
        overlap=overlap,
        origin=origin,
        do_viz_matches=True,
        do_viz_tiles=True,
        save_dir=outdir / str(tile_selection).split(".")[1],
        geometric_verification=GeometricVerification.PYDEGENSAC,
        threshold=1,
        confidence=0.9999,
    )

    tile_selection = TileSelection.GRID
    matcher.match(
        img0,
        img1,
        tile_selection=tile_selection,
        grid=grid,
        overlap=overlap,
        origin=origin,
        do_viz_matches=True,
        do_viz_tiles=True,
        save_dir=outdir / str(tile_selection).split(".")[1],
        geometric_verification=GeometricVerification.PYDEGENSAC,
        threshold=1,
        confidence=0.9999,
    )

    tile_selection = TileSelection.EXHAUSTIVE
    matcher.match(
        img0,
        img1,
        tile_selection=tile_selection,
        grid=grid,
        overlap=overlap,
        origin=origin,
        do_viz_matches=True,
        do_viz_tiles=True,
        save_dir=outdir / str(tile_selection).split(".")[1],
        geometric_verification=GeometricVerification.PYDEGENSAC,
        threshold=1,
        confidence=0.9999,
    )

    print("Matching succeded.")

    # -------------------- OLD --------------------

    # Test Superpoint class
    # device = torch.device("cuda")
    # superpoint_detector = SuperPoint_features(cfg.matching.max_keypoints)
    # features0 = superpoint_detector.detect_and_describe(img0)
    # features1 = superpoint_detector.detect_and_describe(img1)

    # matching = Matching().eval().to(device)

    # _, tens0, _ = read_image(img0, device, [2400], 0, True)
    # _, tens1, _ = read_image(img1, device, [2400], 0, True)

    # data = {
    #     "image0": tens0,
    #     "image1": tens1,
    # }
    # data = {**data, **{k + "0": v for k, v in features0.items()}}
    # data = {**data, **{k + "1": v for k, v in features1.items()}}

    # pred = matching(data)


# class SuperPoint_features:
#     def __init__(
#         self,
#         max_keypoints: int = 2048,
#         keypoint_threshold: float = 0.0001,
#         use_cuda: bool = True,
#         #  weights_path: Path = MODEL_WEIGHTS_PATH,
#     ) -> None:
#         """Configures the object.
#         Args:
#             max_keypoints: max keypoints to detect in an image.
#             keypoint_threshold: threshold for keypoints detection
#             use_cuda (optional): flag controlling the use of GPUs via CUDA. Defaults to True.
#             # weights_path (optional): Path to the model weights. Defaults to MODEL_WEIGHT_PATH.
#         """
#         self._use_cuda = use_cuda and torch.cuda.is_available()
#         self._config = {
#             "superpoint": {
#                 "nms_radius": NMS_RADIUS,
#                 "keypoint_threshold": keypoint_threshold,
#                 "max_keypoints": max_keypoints,
#             },
#         }

#     def detect_and_describe(self, im_path: Path):
#         """Jointly generate keypoint detections and their associated descriptors from a single image."""
#         # TODO(ayushbaid): fix inference issue #110
#         device = torch.device("cuda" if self._use_cuda else "cpu")
#         model = SuperPoint(self._config).eval().to(device)

#         # Read image and transform to tensor
#         image, image_tensor, _ = read_image(im_path, device, [2400], 0, True)

#         # Compute features.
#         with torch.no_grad():
#             model_results = model({"image": image_tensor})
#         torch.cuda.empty_cache()

#         return model_results

# keypoints = model_results["keypoints"][0].detach().cpu().numpy()
# scores = model_results["scores"][0].detach().cpu().numpy()
# descriptors = model_results["descriptors"][0].detach().cpu().numpy()

# features = Features
# features.append_features(
#     {
#         "kpts": keypoints,
#         "descr": descriptors,
#         "score": scores,
#     }
# )

# return features
