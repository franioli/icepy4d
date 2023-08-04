import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict

from icepy4d.matching.enums import GeometricVerification, Quality, TileSelection
from icepy4d.matching.geometric_verification import geometric_verification
from icepy4d.utils import AverageTimer, timeit

matplotlib.use("TkAgg")


@dataclass
class FeaturesBase:
    keypoints: np.ndarray
    descriptors: np.ndarray = None
    scores: np.ndarray = None


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

        # initialize additional variable members for storing matched
        # keypoints descriptors and scores
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

    def _match_images(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """Matches keypoints and descriptors in two given images (no
        matter if they are tiles or full-res images) using the
        SuperGlue algorithm.

        This method takes in two images as Numpy arrays, and returns
        the matches between keypoints
        and descriptors in those images using the SuperGlue algorithm.

        Args:
            image0 (np.ndarray): the first image to match, as Numpy array
            image1 (np.ndarray): the second image to match, as Numpy array

        Returns:
            Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]: a
            tuple containing the features of the first image, the features
            of the second image, the matches between them and the match
            confidence.
        """
        raise NotImplementedError(
            "Subclasses must implement _match_full_images() method."
        )

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
        try:
            self._store_features(features0, features1, matches0)
            self._mconf = mconf
        except Exception as e:
            logging.error(
                f"""Error storing matches: {e}. 
                Implement your own _store_features() method if the
                output of your matcher is different from FeaturesBase."""
            )
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
            try:
                self._viz_matches_mpl(
                    image0, image1, self._mkpts0, self._mkpts1, save_dir / "matches.png"
                )
            except Exception as e:
                logging.error(
                    f"Error visualizing matches with OpenCV: {e}. Fallback to maplotlib."
                )
                self._viz_matches_mpl(
                    image0,
                    image1,
                    self._mkpts0,
                    self._mkpts1,
                    save_dir / "matches.png",
                    hide_fig=True,
                )

        self.timer.print("Matching")

        return True

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

    def _store_features(
        self,
        features0: FeaturesBase,
        features1: FeaturesBase,
        matches0: np.ndarray,
        force_overwrite: bool = True,
    ) -> bool:
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
                return False
            else:
                logging.warning("Matches already stored. Overwrite them")

        valid = matches0 > -1
        self._valid = valid
        idx1 = matches0[valid]
        self._mkpts0 = features0.keypoints[valid]
        self._mkpts1 = features1.keypoints[idx1]
        if features0.descriptors is not None:
            self._descriptors0 = features0.descriptors[:, valid]
            self._descriptors1 = features1.descriptors[:, idx1]
        if features0.scores is not None:
            self._scores0 = features0.scores[valid]
            self._scores1 = features1.scores[idx1]

        return True

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
