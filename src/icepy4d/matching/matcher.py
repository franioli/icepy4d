"""SuperGlue matcher implementation
The network was proposed in 'SuperGlue: Learning Feature Matching with Graph Neural Networks' and is implemented by wrapping over author's source-code.
Note: the pretrained model only supports SuperPoint detections currently.
References:
- http://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
- https://github.com/magicleap/SuperGluePretrainedNetwork
"""

import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import kornia as K
import kornia.feature as KF
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict

from icepy4d.thirdparty.SuperGlue.models.matching import Matching
from icepy4d.thirdparty.SuperGlue.models.utils import make_matching_plot
from icepy4d.utils import AverageTimer, timeit

from .tiler import Tiler
from .enums import GeometricVerification, Quality, TileSelection
from .geometric_verification import geometric_verification

matplotlib.use("TkAgg")


# TODO: use KORNIA for image tiling

# SuperPoint and SuperGlue default parameters
NMS_RADIUS = 3
SUPERGLUE_DESC_DIM = 256
SINKHORN_ITERATIONS = 10


@dataclass
class FeaturesBase:
    keypoints: np.ndarray
    descriptors: np.ndarray = None
    scores: np.ndarray = None


def check_dict_keys(dict: dict, keys: List[str]):
    missing_keys = [key for key in keys if key not in dict]
    if missing_keys:
        raise KeyError(
            f"Missing required keys: {', '.join(missing_keys)} Matcher option dictionary"
        )


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
        """
        Base class for matchers.

        Args:
            opt (dict): Options for the matcher.

        Raises:
            TypeError: If `opt` is not a dictionary.
        """
        if not isinstance(opt, dict):
            raise TypeError("opt must be a dictionary")
        self._opt = edict(opt)
        self._device = (
            "cuda" if torch.cuda.is_available() and not opt.get("force_cpu") else "cpu"
        )
        logging.info(f"Running inference on device {self._device}")

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
        self._mkpts0 = None
        self._mkpts1 = None
        self._descriptors0 = None
        self._descriptors1 = None
        self._scores0 = None
        self._scores1 = None
        self._mconf = None

    @property
    def device(self):
        return self._device

    @property
    def mkpts0(self):
        return self._mkpts0

    @property
    def mkpts1(self):
        return self._mkpts1

    @property
    def descriptors0(self):
        return self._descriptors0

    @property
    def descriptors1(self):
        return self._descriptors1

    @property
    def scores0(self):
        return self._scores0

    @property
    def scores1(self):
        return self._scores1

    @property
    def mconf(self):
        return self._mconf

    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        quality: Quality = Quality.HIGH,
        tile_selection: TileSelection = TileSelection.NONE,
        **kwargs,
    ) -> bool:
        """
        Matches images and performs geometric verification.

        Args:
            image0: The first input image as a NumPy array.
            image1: The second input image as a NumPy array.
            quality: The quality level for resizing images (default: Quality.HIGH).
            tile_selection: The method for selecting tiles for matching (default: TileSelection.NONE).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            A boolean indicating the success of the matching process.

        """
        raise NotImplementedError("Subclasses must implement match() method.")

    def _match_images(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """Matches keypoints and descriptors in two given images (no matter if they are tiles or full-res images) using the SuperGlue algorithm.

        This method takes in two images as Numpy arrays, and returns the matches between keypoints
        and descriptors in those images using the SuperGlue algorithm.

        Args:
            image0 (np.ndarray): the first image to match, as Numpy array
            image1 (np.ndarray): the second image to match, as Numpy array

        Returns:
            Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]: a tuple containing the features of the first image, the features of the second image, the matches between them and the match confidence.
        """
        raise NotImplementedError(
            "Subclasses must implement _match_full_images() method."
        )

    def _tile_selection(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        t0_lims: dict[int, np.ndarray],
        t1_lims: dict[int, np.ndarray],
        method: TileSelection = TileSelection.PRESELECTION,
    ) -> List[Tuple[int, int]]:
        """
        Selects tile pairs for matching based on the specified method.

        Args:
            image0 (np.ndarray): The first image.
            image1 (np.ndarray): The second image.
            t0_lims (dict[int, np.ndarray]): The limits of tiles in image0.
            t1_lims (dict[int, np.ndarray]): The limits of tiles in image1.
            method (TileSelection, optional): The tile selection method. Defaults to TileSelection.PRESELECTION.

        Returns:
            List[Tuple[int, int]]: The selected tile pairs.

        """

        def points_in_rect(points: np.ndarray, rect: np.ndarray) -> np.ndarray:
            logic = np.all(points > rect[:2], axis=1) & np.all(
                points < rect[2:], axis=1
            )
            return logic

        # default parameters
        min_matches_per_tile = 2

        # Select tile selection method
        if method == TileSelection.EXHAUSTIVE:
            # Match all the tiles with all the tiles
            logging.info("Matching tiles exaustively")
            tile_pairs = sorted(product(t0_lims.keys(), t1_lims.keys()))
        elif method == TileSelection.GRID:
            # Match tiles by regular grid
            logging.info("Matching tiles by regular grid")
            tile_pairs = sorted(zip(t0_lims.keys(), t1_lims.keys()))
        elif method == TileSelection.PRESELECTION:
            # Match tiles by preselection running matching on downsampled images
            logging.info("Matching tiles by preselection tile selection")
            if image0.shape[0] > 8000:
                n_down = 4
            if image0.shape[0] > 4000:
                n_down = 3
            elif image0.shape[0] > 2000:
                n_down = 2
            else:
                n_down = 1

            # Run inference on downsampled images
            i0 = deepcopy(image0)
            i1 = deepcopy(image1)
            for _ in range(n_down):
                i0 = cv2.pyrDown(i0)
                i1 = cv2.pyrDown(i1)
            f0, f1, mtc, _ = self._match_images(i0, i1)
            vld = mtc > -1
            kp0 = f0.keypoints[vld]
            kp1 = f1.keypoints[mtc[vld]]
            for _ in range(n_down):
                kp0 *= 2
                kp1 *= 2

            # Select tile pairs where there are enough matches
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

            # Debug...
            # c = "r"
            # s = 5
            # fig, axes = plt.subplots(1, 2)
            # for ax, img, kp in zip(axes, [image0, image1], [kp0, kp1]):
            #     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR))
            #     ax.scatter(kp[:, 0], kp[:, 1], s=s, c=c)
            #     ax.axis("off")
            # for lim0, lim1 in zip(t0_lims.values(), t1_lims.values()):
            #     axes[0].axvline(lim0[0])
            #     axes[0].axhline(lim0[1])
            #     axes[1].axvline(lim1[0])
            #     axes[1].axhline(lim1[1])
            # # axes[1].get_yaxis().set_visible(False)
            # fig.tight_layout()
            # plt.show()
            # fig.savefig("preselection.png")
            # plt.close()

        return tile_pairs

    def _match_tiles(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        tile_selection: TileSelection = TileSelection.PRESELECTION,
        **kwargs,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """
        Matches tiles in two images and returns the features, matches, and confidence.

        Args:
            image0: The first input image as a NumPy array.
            image1: The second input image as a NumPy array.
            tile_selection: The method for selecting tile pairs to match (default: TileSelection.PRESELECTION).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            A tuple containing:
            - features0: FeaturesBase object representing keypoints, descriptors, and scores of image0.
            - features1: FeaturesBase object representing keypoints, descriptors, and scores of image1.
            - matches0: NumPy array with indices of matched keypoints in image0.
            - mconf: NumPy array with confidence scores for the matches.

        """
        raise NotImplementedError("Subclasses must implement _match_tiles() method.")

    def _resize_images(
        self, quality: Quality, image0: np.ndarray, image1: np.ndarray
    ) -> Tuple[np.ndarray]:
        """
        Resize images based on the specified quality.

        Args:
            quality (Quality): The quality level for resizing.
            image0 (np.ndarray): The first image.
            image1 (np.ndarray): The second image.

        Returns:
            Tuple[np.ndarray]: Resized images.

        """
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
        self, quality: Quality, features0: FeaturesBase, features1: FeaturesBase
    ) -> Tuple[FeaturesBase]:
        """
        Resize features based on the specified quality.

        Args:
            quality (Quality): The quality level for resizing.
            features0 (FeaturesBase): The features of the first image.
            features1 (FeaturesBase): The features of the second image.

        Returns:
            Tuple[FeaturesBase]: Resized features.

        """
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

    def _filter_matches_by_mask(self, inlMask: np.ndarray) -> None:
        """
        Filter matches based on the specified mask.

        Args:
            inlMask (np.ndarray): The mask to filter matches.
        """
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
        s = kwargs.get("s", 2)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(cv2.cvtColor(image0, cv2.COLOR_BGR2RGB))
        ax[0].scatter(kpts0[:, 0], kpts0[:, 1], s=s, c=colors[0])
        ax[1].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        ax[1].scatter(kpts1[:, 0], kpts1[:, 1], s=s, c=colors[1])
        if save_path is not None:
            fig.savefig(save_path)
        if hide_fig is False:
            plt.show()
        else:
            plt.close(fig)


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
        opt = self._build_superglue_config(opt)
        super().__init__(opt)

        # Inizialize the Matching object with given configuration
        self.matcher = Matching(self._opt).eval().to(self._device)

    def _build_superglue_config(self, opt: dict) -> dict:
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

    def _frame2tensor(self, frame, device):
        return torch.from_numpy(frame / 255.0).float()[None, None].to(device)

    @timeit
    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        quality: Quality = Quality.HIGH,
        tile_selection: TileSelection = TileSelection.NONE,
        **kwargs,
    ) -> bool:
        """
        Matches images and performs geometric verification.

        Args:
            image0: The first input image as a NumPy array.
            image1: The second input image as a NumPy array.
            quality: The quality level for resizing images (default: Quality.HIGH).
            tile_selection: The method for selecting tiles for matching (default: TileSelection.NONE).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            A boolean indicating the success of the matching process.

        """
        self.timer = AverageTimer()

        # Get kwargs
        do_viz_matches = kwargs.get("do_viz_matches", False)
        save_dir = kwargs.get("save_dir", ".")
        gv_method = kwargs.get(
            "geometric_verification", GeometricVerification.PYDEGENSAC
        )
        threshold = kwargs.get("threshold", 1)
        confidence = kwargs.get("confidence", 0.9999)

        # Resize images if needed
        image0_, image1_ = self._resize_images(quality, image0, image1)

        # Perform matching (on tiles or full images)
        if tile_selection == TileSelection.NONE:
            logging.info("Matching full images...")
            features0, features1, matches0, _ = self._match_images(image0_, image1_)

        else:
            logging.info("Matching by tiles...")
            features0, features1, matches0, _ = self._match_tiles(
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

        if do_viz_matches is True:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.viz_matches(image0, image1, save_dir / f"matches.png")

        self.timer.print("Matching")

        return True

    def _match_images(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """Matches keypoints and descriptors in two given images (no matter if they are tiles or full-res images) using the SuperGlue algorithm.

        This method takes in two images as Numpy arrays, and returns the matches between keypoints
        and descriptors in those images using the SuperGlue algorithm.

        Args:
            image0 (np.ndarray): the first image to match, as Numpy array
            image1 (np.ndarray): the second image to match, as Numpy array

        Returns:
            Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]: a tuple containing the features of the first image, the features of the second image, the matches between them and the match confidence.
        """

        if len(image0.shape) > 2:
            image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
        if len(image1.shape) > 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

        tensor0 = self._frame2tensor(image0, self._device)
        tensor1 = self._frame2tensor(image1, self._device)

        with torch.inference_mode():
            pred_tensor = self.matcher({"image0": tensor0, "image1": tensor1})
        pred = {k: v[0].cpu().numpy() for k, v in pred_tensor.items()}

        # Create FeaturesBase objects and matching array
        features0 = FeaturesBase(
            keypoints=pred["keypoints0"],
            descriptors=pred["descriptors0"],
            scores=pred["scores0"],
        )
        features1 = FeaturesBase(
            keypoints=pred["keypoints1"],
            descriptors=pred["descriptors1"],
            scores=pred["scores1"],
        )
        matches0 = pred["matches0"]

        # Create a match confidence array
        valid = matches0 > -1
        mconf = features0.scores[valid]

        logging.info("Matching completed.")

        return features0, features1, matches0, mconf

    def _match_tiles(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        tile_selection: TileSelection = TileSelection.PRESELECTION,
        **kwargs,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """
        Matches tiles in two images and returns the features, matches, and confidence.

        Args:
            image0: The first input image as a NumPy array.
            image1: The second input image as a NumPy array.
            tile_selection: The method for selecting tile pairs to match (default: TileSelection.PRESELECTION).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            A tuple containing:
            - features0: FeaturesBase object representing keypoints, descriptors, and scores of image0.
            - features1: FeaturesBase object representing keypoints, descriptors, and scores of image1.
            - matches0: NumPy array with indices of matched keypoints in image0.
            - mconf: NumPy array with confidence scores for the matches.

        Raises:
            AssertionError: If image0 or image1 is not a NumPy array.

        """
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
            tensor0 = self._(tile0, self._device)
            tensor1 = self._frame2tensor(tile1, self._device)
            with torch.inference_mode():
                pred_tensor = self.matcher({"image0": tensor0, "image1": tensor1})
            pred = {k: v[0].cpu().numpy() for k, v in pred_tensor.items()}

            # Get matches, descriptors and scores
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

            # Append matches to full array
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

            # Visualize matches on tile
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
        features0 = FeaturesBase(
            keypoints=mkpts0_full, descriptors=descriptors0_full, scores=scores0_full
        )
        features1 = FeaturesBase(
            keypoints=mkpts1_full, descriptors=descriptors1_full, scores=scores1_full
        )

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0_full.shape[0])

        # Create a match confidence array
        valid = matches0 > -1
        mconf = features0.scores[valid]

        logging.info("Matching by tile completed.")

        return features0, features1, matches0, mconf

    def _store_features(
        self,
        features0: FeaturesBase,
        features1: FeaturesBase,
        matches0: np.ndarray,
        force_overwrite: bool = True,
    ) -> None:
        """Stores keypoints, descriptors and scores of the matches in the object's members."""

        assert isinstance(
            features0, FeaturesBase
        ), "features0 must be a FeaturesBase object"
        assert isinstance(
            features1, FeaturesBase
        ), "features1 must be a FeaturesBase object"
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


class LOFTRMatcher(ImageMatcherBase):
    def __init__(self, opt: dict = {}) -> None:
        """Initializes a LOFTRMatcher with Kornia object with the given options dictionary."""

        opt = self._build_config(opt)
        super().__init__(opt)

        self.matcher = KF.LoFTR(pretrained="outdoor").to(self.device).eval()

    def _build_config(self, opt: dict) -> dict:
        def_opt = {
            "pretrained": "outdoor",
            "force_cpu": False,
        }
        opt = {**def_opt, **opt}
        required_keys = [
            "pretrained",
            "force_cpu",
        ]
        check_dict_keys(opt, required_keys)

        return opt

    def _img_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = K.image_to_tensor(np.array(image), False).float() / 255.0
        image = K.color.bgr_to_rgb(image.to(self.device))
        if image.shape[1] > 2:
            image = K.color.rgb_to_grayscale(image)
        return image

    @timeit
    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        quality: Quality = Quality.HIGH,
        tile_selection: TileSelection = TileSelection.NONE,
        **kwargs,
    ) -> bool:
        """
        Matches images and performs geometric verification.

        Args:
            image0: The first input image as a NumPy array.
            image1: The second input image as a NumPy array.
            quality: The quality level for resizing images (default: Quality.HIGH).
            tile_selection: The method for selecting tiles for matching (default: TileSelection.NONE).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            A boolean indicating the success of the matching process.

        """
        self.timer = AverageTimer()

        # Get kwargs
        do_viz_matches = kwargs.get("do_viz_matches", False)
        save_dir = kwargs.get("save_dir", ".")
        gv_method = kwargs.get(
            "geometric_verification", GeometricVerification.PYDEGENSAC
        )
        threshold = kwargs.get("threshold", 1)
        confidence = kwargs.get("confidence", 0.9999)

        # Resize images if needed
        image0_, image1_ = self._resize_images(quality, image0, image1)

        # Perform matching (on tiles or full images)
        if tile_selection == TileSelection.NONE:
            logging.info("Matching full images...")
            features0, features1, matches0, mconf = self._match_images(image0_, image1_)

        else:
            logging.info("Matching by tiles...")
            features0, features1, matches0, mconf = self._match_tiles(
                image0_, image1_, tile_selection, **kwargs
            )

        # Retrieve original image coordinates if matching was performed on up/down-sampled images
        features0, features1 = self._resize_features(quality, features0, features1)

        # Store features as class members
        self._mkpts0 = features0.keypoints
        self._mkpts1 = features1.keypoints
        self._mconf = mconf
        self.timer.update("matching")
        logging.info("Matching done!")

        if do_viz_matches is True:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self._viz_matches_mpl(
                image0, image1, self._mkpts0, self._mkpts1, save_dir / f"matches.png"
            )

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

        if do_viz_matches is True:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self._viz_matches_mpl(
                image0,
                image1,
                self._mkpts0,
                self._mkpts1,
                save_dir / f"matches_valid.png",
            )

        self.timer.print("Matching")

        return True

    def _match_images(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """Matches keypoints and descriptors in two given images (no matter if they are tiles or full-res images) using the SuperGlue algorithm.

        This method takes in two images as Numpy arrays, and returns the matches between keypoints
        and descriptors in those images using the SuperGlue algorithm.

        Args:
            image0 (np.ndarray): the first image to match, as Numpy array
            image1 (np.ndarray): the second image to match, as Numpy array

        Returns:
            Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]: a tuple containing the features of the first image, the features of the second image, the matches between them and the match confidence.
        """

        # Covert images to tensor
        timg0_ = self._img_to_tensor(image0)
        timg1_ = self._img_to_tensor(image1)

        # Run inference
        with torch.inference_mode():
            input_dict = {"image0": timg0_, "image1": timg1_}
            correspondences = self.matcher(input_dict)

        # Get matches and build features
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        features0 = FeaturesBase(keypoints=mkpts0)
        features1 = FeaturesBase(keypoints=mkpts1)

        # Get match confidence
        mconf = correspondences["confidence"].cpu().numpy()

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0.shape[0])

        return features0, features1, matches0, mconf

    def _match_tiles(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        tile_selection: TileSelection = TileSelection.PRESELECTION,
        **kwargs,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """
        Matches tiles in two images and returns the features, matches, and confidence.

        Args:
            image0: The first input image as a NumPy array.
            image1: The second input image as a NumPy array.
            tile_selection: The method for selecting tile pairs to match (default: TileSelection.PRESELECTION).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            A tuple containing:
            - features0: FeaturesBase object representing keypoints, descriptors, and scores of image0.
            - features1: FeaturesBase object representing keypoints, descriptors, and scores of image1.
            - matches0: NumPy array with indices of matched keypoints in image0.
            - mconf: NumPy array with confidence scores for the matches.

        """
        # Get kwargs
        grid = kwargs.get("grid", [1, 1])
        overlap = kwargs.get("overlap", 0)
        origin = kwargs.get("origin", [0, 0])
        do_viz_tiles = kwargs.get("do_viz_tiles", False)

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
        conf_full = np.array([], dtype=np.float32)

        # Match each tile pair
        for tidx0, tidx1 in tile_pairs:
            logging.info(f" - Matching tile pair ({tidx0}, {tidx1})")

            lim0 = t0_lims[tidx0]
            lim1 = t1_lims[tidx1]
            tile0 = self._tiler.extract_patch(image0, lim0)
            tile1 = self._tiler.extract_patch(image1, lim1)

            # Covert patch to tensor
            timg0_ = self._img_to_tensor(tile0)
            timg1_ = self._img_to_tensor(tile1)

            # Run inference
            with torch.inference_mode():
                input_dict = {"image0": timg0_, "image1": timg1_}
                correspondences = self.matcher(input_dict)

            # Get matches and build features
            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()

            # Get match confidence
            conf = correspondences["confidence"].cpu().numpy()

            # Append to full arrays
            mkpts0_full = np.vstack(
                (mkpts0_full, mkpts0 + np.array(lim0[0:2]).astype("float32"))
            )
            mkpts1_full = np.vstack(
                (mkpts1_full, mkpts1 + np.array(lim1[0:2]).astype("float32"))
            )
            conf_full = np.concatenate((conf_full, conf))

            # Plot matches on tile
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

        # Select uniue features on image 0, on rounded coordinates
        decimals = 1
        _, unique_idx = np.unique(
            np.round(mkpts0_full, decimals), axis=0, return_index=True
        )
        mkpts0_full = mkpts0_full[unique_idx]
        mkpts1_full = mkpts1_full[unique_idx]
        conf_full = conf_full[unique_idx]

        # Create features
        features0 = FeaturesBase(keypoints=mkpts0_full)
        features1 = FeaturesBase(keypoints=mkpts1_full)

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0_full.shape[0])

        logging.info("Matching by tile completed.")

        return features0, features1, matches0, conf_full


class LightGlueMatcher(ImageMatcherBase):
    def __init__(self, opt: dict = {}) -> None:
        """Initializes a LOFTRMatcher with Kornia object with the given options dictionary."""

        opt = self._build_config(opt)
        super().__init__(opt)

        self.matcher = KF.LoFTR(pretrained="outdoor").to(self.device).eval()

    def _img_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = K.image_to_tensor(np.array(image), False).float() / 255.0
        image = K.color.bgr_to_rgb(image.to(self.device))
        if image.shape[1] > 2:
            image = K.color.rgb_to_grayscale(image)
        return image

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

        # Resize images if needed
        image0_, image1_ = self._resize_images(quality, image0, image1)

        # Perform matching (on tiles or full images)
        if tile_selection == TileSelection.NONE:
            logging.info("Matching full images...")
            features0, features1, matches0, mconf = self._match_images(image0_, image1_)

        else:
            logging.info("Matching by tiles...")
            features0, features1, matches0, mconf = self._match_tiles(
                image0_, image1_, tile_selection, **kwargs
            )

        # Retrieve original image coordinates if matching was performed on up/down-sampled images
        features0, features1 = self._resize_features(quality, features0, features1)

        # Store features as class members
        self._mkpts0 = features0.keypoints
        self._mkpts1 = features1.keypoints
        self._mconf = mconf
        self.timer.update("matching")
        logging.info("Matching done!")

        if do_viz_matches is True:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self._viz_matches_mpl(
                image0, image1, self._mkpts0, self._mkpts1, save_dir / f"matches.png"
            )

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

        if do_viz_matches is True:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self._viz_matches_mpl(
                image0,
                image1,
                self._mkpts0,
                self._mkpts1,
                save_dir / f"matches_valid.png",
            )

        self.timer.print("Matching")

    def _match_images(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """Matches keypoints and descriptors in two given images (no matter if they are tiles or full-res images) using the SuperGlue algorithm.

        This method takes in two images as Numpy arrays, and returns the matches between keypoints
        and descriptors in those images using the SuperGlue algorithm.

        Args:
            image0 (np.ndarray): the first image to match, as Numpy array
            image1 (np.ndarray): the second image to match, as Numpy array

        Returns:
            Tuple[FeaturesBase, FeaturesBase, np.ndarray]: a tuple containing the features of the first image, the features of the second image, and the matches between them
        """

        # Covert images to tensor
        timg0_ = self._img_to_tensor(image0)
        timg1_ = self._img_to_tensor(image1)

        # Run inference
        with torch.inference_mode():
            input_dict = {"image0": timg0_, "image1": timg1_}
            correspondences = self.matcher(input_dict)

        # Get matches and build features
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        features0 = FeaturesBase(keypoints=mkpts0)
        features1 = FeaturesBase(keypoints=mkpts1)

        # Get match confidence
        mconf = correspondences["confidence"].cpu().numpy()

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0.shape[0])

        return features0, features1, matches0, mconf

    def _tile_selection(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        t0_lims: dict[int, np.ndarray],
        t1_lims: dict[int, np.ndarray],
        method: TileSelection = TileSelection.PRESELECTION,
    ) -> List[Tuple[int, int]]:
        """
        Selects tile pairs for matching based on the specified method.

        Args:
            image0 (np.ndarray): The first image.
            image1 (np.ndarray): The second image.
            t0_lims (dict[int, np.ndarray]): The limits of tiles in image0.
            t1_lims (dict[int, np.ndarray]): The limits of tiles in image1.
            method (TileSelection, optional): The tile selection method. Defaults to TileSelection.PRESELECTION.

        Returns:
            List[Tuple[int, int]]: The selected tile pairs.

        """

        def points_in_rect(points: np.ndarray, rect: np.ndarray) -> np.ndarray:
            logic = np.all(points > rect[:2], axis=1) & np.all(
                points < rect[2:], axis=1
            )
            return logic

        # default parameters
        min_matches_per_tile = 2

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
            f0, f1, mtc, _ = self._match_images(i0, i1)
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
        # Get kwargs
        grid = kwargs.get("grid", [1, 1])
        overlap = kwargs.get("overlap", 0)
        origin = kwargs.get("origin", [0, 0])
        do_viz_tiles = kwargs.get("do_viz_tiles", False)

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
        conf_full = np.array([], dtype=np.float32)

        # Match each tile pair
        for tidx0, tidx1 in tile_pairs:
            logging.info(f" - Matching tile pair ({tidx0}, {tidx1})")

            lim0 = t0_lims[tidx0]
            lim1 = t1_lims[tidx1]
            tile0 = self._tiler.extract_patch(image0, lim0)
            tile1 = self._tiler.extract_patch(image1, lim1)

            # Covert patch to tensor
            timg0_ = self._img_to_tensor(tile0)
            timg1_ = self._img_to_tensor(tile1)

            # Run inference
            with torch.inference_mode():
                input_dict = {"image0": timg0_, "image1": timg1_}
                correspondences = self.matcher(input_dict)

            # Get matches and build features
            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()

            # Get match confidence
            conf = correspondences["confidence"].cpu().numpy()

            # Append to full arrays
            mkpts0_full = np.vstack(
                (mkpts0_full, mkpts0 + np.array(lim0[0:2]).astype("float32"))
            )
            mkpts1_full = np.vstack(
                (mkpts1_full, mkpts1 + np.array(lim1[0:2]).astype("float32"))
            )
            conf_full = np.concatenate((conf_full, conf))

            # Plot matches on tile
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

        # Select uniue features on image 0, on rounded coordinates
        decimals = 1
        _, unique_idx = np.unique(
            np.round(mkpts0_full, decimals), axis=0, return_index=True
        )
        mkpts0_full = mkpts0_full[unique_idx]
        mkpts1_full = mkpts1_full[unique_idx]
        conf_full = conf_full[unique_idx]

        # Create features
        features0 = FeaturesBase(keypoints=mkpts0_full)
        features1 = FeaturesBase(keypoints=mkpts1_full)

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0_full.shape[0])

        logging.info("Matching by tile completed.")

        return features0, features1, matches0, conf_full


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

    # Test LOFTR
    # Subsample for testing
    # img0 = cv2.pyrDown(cv2.pyrDown(img0))
    # img1 = cv2.pyrDown(cv2.pyrDown(img1))
    grid = [5, 4]
    overlap = 100
    origin = [0, 0]
    matcher = LOFTRMatcher()
    matcher.match(
        img0,
        img1,
        quality=Quality.HIGH,
        tile_selection=TileSelection.PRESELECTION,
        grid=grid,
        overlap=overlap,
        origin=origin,
        do_viz_matches=True,
        do_viz_tiles=True,
        save_dir=outdir / "LOFTR",
        geometric_verification=GeometricVerification.PYDEGENSAC,
        threshold=2,
        confidence=0.9999,
    )

    # SuperGlue
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
        quality=Quality.HIGH,
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

    # tile_selection = TileSelection.EXHAUSTIVE
    # matcher.match(
    #     img0,
    #     img1,
    #     tile_selection=tile_selection,
    #     grid=grid,
    #     overlap=overlap,
    #     origin=origin,
    #     do_viz_matches=True,
    #     do_viz_tiles=True,
    #     save_dir=outdir / str(tile_selection).split(".")[1],
    #     geometric_verification=GeometricVerification.PYDEGENSAC,
    #     threshold=1,
    #     confidence=0.9999,
    # )

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
