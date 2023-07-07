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
from itertools import product
from pathlib import Path
from typing import Dict, List, Union

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict

from icepy4d.thirdparty.SuperGlue.models.matching import Matching
from icepy4d.thirdparty.SuperGlue.models.utils import make_matching_plot
from icepy4d.utils import AverageTimer

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


class Tiler:
    """
    Class for dividing an image into tiles.
    """

    def __init__(
        self,
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
    method: str = "pydegensac",
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

    TODO: allow input parameters for both pydegensac and opencv.

    """

    available_methods = ["pydegensac", "opencv"]
    if method not in available_methods:
        logging.error(
            f"Input method '{method}' not availble. Method must be one of {available_methods}. Using opencv instead."
        )
        method = "opencv"

    if method == "pydegensac":
        try:
            pydegensac = importlib.import_module("pydegensac")
            fallback = False
        except:
            logging.error(
                "Pydegensac not available. Using MAGSAC++ (OpenCV) for geometric verification."
            )
            fallback = True

    if method == "pydegensac" and not fallback:
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

    if method == "opencv" or fallback:
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

    # self.mkpts0 = self.mkpts0[self._inlMask, :]
    # self.mkpts1 = self.mkpts1[self._inlMask, :]
    # if self.matches0 is not None:
    #     self.match_conf = self.match_conf[self._inlMask]
    # if self.descriptors0 is not None:
    #     self.descriptors0 = self.descriptors0[:, self._inlMask]
    # if self.descriptors1 is not None:
    #     self.descriptors1 = self.descriptors1[:, self._inlMask]
    # if self.scores0 is not None:
    #     self.scores0 = self.scores0[self._inlMask]
    # if self.scores1 is not None:
    #     self.scores1 = self.scores1[self._inlMask]
    # return {0: mkpts0, 1: mkpts1}


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
    def _match_full_images(self):
        pass

    @abstractmethod
    def _match_tiles(self):
        pass

    @abstractmethod
    def geometric_verification(self):
        pass

    # @abstractmethod
    # def _generate_tiles(self):
    #     pass

    # @abstractmethod
    # def _combine_tiles(self):
    #     pass


class ImageMatcherBase(ImageMatcherABC):
    def __init__(self, opt: dict = {}) -> None:
        """Base class for matchers"""
        if not isinstance(opt, dict):
            raise TypeError("opt must be a dictionary")
        self._opt = edict(opt)

        # self.features0 and self.features1 are dictionaries containing keypoints, descriptors and scores for each image and it is set by the method self.match()
        self._features0 = {}  # Keypoints, descriptors and scores for image 0
        self._features1 = {}  # Keypoints, descriptors and scores for image 1
        self._matches0 = None  # Index of the matches from image 0 to image 1

        self._device = (
            "cuda" if torch.cuda.is_available() and not opt.get("force_cpu") else "cpu"
        )
        logging.info(f"Running inference on device {self._device}")

    @property
    def device(self):
        return self._device

    @property
    def features0(self):
        return self._features0

    @features0.setter
    def features0(self, value):
        """
        features0 _summary_

        Args:
            value (_type_): _description_

        TODO: add checks on input dictionary. At least keypoints must be present
        """
        assert isinstance(value, dict), "features0 must be a dictionary"
        assert (
            "keypoints" in value.keys()
        ), "At least 'keypoints' must be present in features0"
        self._features0 = value

    @property
    def features1(self):
        return self._features1

    @features1.setter
    def features1(self, value):
        assert isinstance(value, dict), "features1 must be a dictionary"
        assert (
            "keypoints" in value.keys()
        ), "At least 'keypoints' must be present in features1"
        self._features1 = value

    @property
    def matches0(self):
        return self._matches0

    @matches0.setter
    def matches0(self, value):
        """Store the matching array from image 0 to image 1. It contains the index of the match on image 1 for each keypoint on image 0
        TODO: add other checks on input array
        """

        if not isinstance(value, np.ndarray):
            raise TypeError("matches0 must be a NumPy array")
        self._matches0 = value

    def match(self):
        """
        Matches keypoints between two images. This method must be implemented by the child class.
        """
        raise NotImplementedError("Subclasses must implement match() method.")

    def _match_full_images(self):
        raise NotImplementedError(
            "Subclasses must implement _match_full_images() method."
        )

    def _match_tiles(self):
        raise NotImplementedError("Subclasses must implement _match_tiles() method.")

    def geometric_verification(self):
        raise NotImplementedError(
            "Subclasses must implement geometric_verification() method."
        )

    def _mask_features(self, features: dict, mask: np.ndarray) -> dict:
        raise NotImplementedError("Subclasses must implement _mask_features() method.")


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

    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        match_by_tile: bool = False,
        **kwargs,
    ):
        self.timer = AverageTimer()

        # Inizialize the Matching object with given configuration
        self.matcher = Matching(self._opt).eval().to(self._device)

        # Perform matching (on tiles or full images)
        if match_by_tile is True:
            logging.info("Matching by tiles...")
            self._match_tiles(image0, image1, **kwargs)
        else:
            logging.info("Matching full images...")
            self._match_full_images(image0, image1)
            self.viz_matches(image0, image1, f"test_full.png")

        self.timer.update("matching")
        logging.info("Matching done!")

        # Perform geometric verification
        logging.info("Performing geometric verification...")
        method = kwargs.get("method", "pydegensac")
        threshold = kwargs.get("threshold", 1)
        confidence = kwargs.get("confidence", 0.9999)
        if self._mkpts0 is None or self._mkpts1 is None:
            logging.warning(
                "mkpts0 and mkpts1 are not save as member of the class. Trying to get them from the matched features..."
            )
            try:
                self._exapand_and_store_matches()
            except AttributeError as e:
                logging.error(
                    "Matched keypoints and descriptors are not available. Peraphs matching was failed."
                )
                raise e

        F, inlMask = geometric_verification(
            self._mkpts0,
            self._mkpts1,
            method=method,
            confidence=confidence,
            threshold=threshold,
        )
        self._F = F
        self._filter_matches_by_mask(inlMask)
        self.timer.update("geometric_verification")
        logging.info("Geometric verification done.")

        self.timer.print("Matching")

    def _match_tiles(self, image0, image1, **kwargs):
        grid = kwargs.get("grid", [1, 1])
        overlap = kwargs.get("overlap", 0)
        origin = kwargs.get("origin", [0, 0])

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
        exaustive = kwargs.get("exaustive", False)
        if exaustive is True:
            logging.info("Matching tiles exaustively")
            tile_pairs = sorted(product(t0_lims.keys(), t1_lims.keys()))
        else:
            logging.info("Matching tiles by regular grid")
            tile_pairs = sorted(zip(t0_lims.keys(), t1_lims.keys()))

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
                self._pred_tensor = self.matcher({"image0": tensor0, "image1": tensor1})
            pred = {k: v[0].cpu().numpy() for k, v in self._pred_tensor.items()}

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

            do_viz_tile = kwargs.get("do_viz_tile", False)
            if do_viz_tile is True:
                self.viz_matches(tile0, tile1, f"test_{tidx0-tidx1}.png")

        logging.info("Restoring full image coordinates of matches...")
        # Restore original image coordinates (not cropped) and run PyDegensac
        mkpts0_full = mkpts0_full + np.array(t0_origin).astype("float32")
        mkpts1_full = mkpts1_full + np.array(t1_origin).astype("float32")

        # Store features dictionary
        self._features0 = {
            "keypoints": mkpts0_full,
            "descriptors": descriptors0_full,
            "scores": scores0_full,
        }
        self._features1 = {
            "keypoints": mkpts1_full,
            "descriptors": descriptors1_full,
            "scores": scores1_full,
        }

        # Create a 1-to-1 matching array
        self._matches0 = np.arange(mkpts0_full.shape[0])

        # Expand and store matches
        self._exapand_and_store_matches(
            self._features0, self._features1, self._matches0, force_overwrite=True
        )

        logging.info("Matching by tile completed.")

    def _match_full_images(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
    ) -> dict:
        """Matches keypoints and descriptors in two given images (no matter if they are tiles or full-res images) using the SuperGlue algorithm.

        This method takes in two images as Numpy arrays, and returns the matches between keypoints
        and descriptors in those images using the SuperGlue algorithm.

        Args:
            image0 (np.ndarray): the first image to match, as Numpy array
            image1 (np.ndarray): the second image to match, as Numpy array

        Returns:
            dict
        """

        if len(image0.shape) > 2:
            image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
        if len(image1.shape) > 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

        tensor0 = frame2tensor(image0, self._device)
        tensor1 = frame2tensor(image1, self._device)

        with torch.inference_mode():
            self._pred_tensor = self.matcher({"image0": tensor0, "image1": tensor1})
        pred = {k: v[0].cpu().numpy() for k, v in self._pred_tensor.items()}

        self._features0 = {
            "keypoints": pred["keypoints0"],
            "descriptors": pred["descriptors0"],
            "scores": pred["scores0"],
        }
        self._features1 = {
            "keypoints": pred["keypoints1"],
            "descriptors": pred["descriptors1"],
            "scores": pred["scores1"],
        }
        self._matches0 = pred["matches0"]

        # valid_idx_0 = self._matches0 > -1
        # valid_idx_1 = self._matches0[valid_idx_0]
        # mkpts0 = self._features0.get("keypoints")[valid_idx_0]
        # mkpts1 = self._features1.get("keypoints")[valid_idx_1]
        # descriptors0 = self._features0.get("descriptors")[:, valid_idx_0]
        # descriptors1 = self._features1.get("descriptors")[:, valid_idx_1]
        # scores0 = self._features0.get("scores")[valid_idx_0]
        # scores1 = self._features1.get("scores")[valid_idx_1]
        # match_conf = pred["matching_scores0"][valid_idx_0]

        self._exapand_and_store_matches(
            self._features0, self._features1, self._matches0, force_overwrite=False
        )

        return {0: self._mkpts0, 1: self._mkpts1}

    def _exapand_and_store_matches(
        self,
        features0: dict,
        features1: dict,
        matches0: dict,
        force_overwrite: bool = False,
    ):
        """Stores keypoints, descriptors and scores of the matches in the object's members."""

        assert "keypoints" in features0.keys(), "No keypoints found in features0"
        assert "keypoints" in features1.keys(), "No keypoints found in features1"

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
        self._mkpts0 = features0.get("keypoints")[valid]
        self._mkpts1 = features1.get("keypoints")[idx1]
        self._descriptors0 = features0.get("descriptors")[:, valid]
        self._descriptors1 = features1.get("descriptors")[:, idx1]
        self._scores0 = features0.get("scores")[valid]
        self._scores1 = features1.get("scores")[idx1]
        self._mconf = features0.get("scores")[valid]

    def _filter_matches_by_mask(self, inlMask: np.ndarray):
        self._mkpts0 = self._mkpts0[inlMask, :]
        self._mkpts1 = self._mkpts1[inlMask, :]
        if self._matches0 is not None:
            self._match_conf = self._mconf[inlMask]
        if self._descriptors0 is not None:
            self._descriptors0 = self._descriptors0[:, inlMask]
        if self._descriptors1 is not None:
            self._descriptors1 = self._descriptors1[:, inlMask]
        if self._scores0 is not None:
            self._scores0 = self._scores0[inlMask]
        if self._scores1 is not None:
            self._scores1 = self._scores1[inlMask]

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
        - fast_viz (bool): Whether to use fast visualization method.
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

    im_path0 = assset_path / "img/cam1/IMG_2637.jpg"
    im_path1 = assset_path / "img/cam2/IMG_1112.jpg"
    img0 = cv2.imread(str(im_path0))
    img1 = cv2.imread(str(im_path1))

    suerglue_cfg = {
        "weights": "outdoor",
        "keypoint_threshold": 0.001,
        "max_keypoints": 4096,
        "match_threshold": 0.3,
        "force_cpu": False,
    }
    matcher = SuperGlueMatcher(suerglue_cfg)
    # mkpts = matcher.match(img0, img1, match_by_tile=True)
    mkpts = matcher.match(
        img0,
        img1,
        match_by_tile=True,
        grid=[2, 2],
        overlap=50,
        origin=[0, 0],
        do_viz_tiles=True,
    )

    # mkpts = matcher.geometric_verification(
    #     method="ransac",
    #     threshold=1,
    #     confidence=0.99,
    #     symmetric_error_check=False,
    # )
    matcher.viz_matches(img0, img1, assset_path / "test.png")

    # os.remove(assset_path / "test.png")

    print("Matching succeded.")

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


# OLD CODE

# old proprties of the class

# @property
# def keypoints0(self):
#     return self._features0.get("keypoints")[self._valid_idx_0]

# @property
# def keypoints1(self):
#     return self._features1.get("keypoints")[self._valid_idx_1]

# @property
# def descriptors0(self):
#     return self._features0.get("descriptors")[:, self._valid_idx_0]

# @property
# def descriptors1(self):
#     return self._features1.get("descriptors")[:, self._valid_idx_1]

# @property
# def scores0(self):
#     return self._features0.get("scores")[self._valid_idx_0]

# @property
# def scores1(self):
#     return self._features1.get("scores")[self._valid_idx_1]

# @property
# def mkpts0(self):
#     "Same as self.keypoints0 Kept for backward compatibility"
#     return self.keypoints0

# @property
# def mkpts1(self):
#     "Same as self.keypoints1 Kept for backward compatibility"
#     return self.keypoints1


# Proprierties for each element of feautures dict

# @property
# def mkpts0(self):
#     return self._mkpts0

# @mkpts0.setter
# def mkpts0(self, value):
#     self._mkpts0 = value

# @property
# def mkpts1(self):
#     return self._mkpts1

# @mkpts1.setter
# def mkpts1(self, value):
#     self._mkpts1 = value

# @property
# def descriptors0(self):
#     return self._descriptors0

# @descriptors0.setter
# def descriptors0(self, value):
#     self._descriptors0 = value

# @property
# def descriptors1(self):
#     return self._descriptors1

# @descriptors1.setter
# def descriptors1(self, value):
#     self._descriptors1 = value

# @property
# def scores0(self):
#     return self._scores0

# @scores0.setter
# def scores0(self, value):
#     self._scores0 = value

# @property
# def scores1(self):
#     return self._scores1

# @scores1.setter
# def scores1(self, value):
#     self._scores1 = value


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
