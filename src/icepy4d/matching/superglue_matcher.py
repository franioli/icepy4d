"""SuperGlue matcher implementation
The network was proposed in 'SuperGlue: Learning Feature Matching with Graph Neural Networks' and is implemented by wrapping over author's source-code.
Note: the pretrained model only supports SuperPoint detections currently.
References:
- http://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
- https://github.com/magicleap/SuperGluePretrainedNetwork
"""

import importlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from easydict import EasyDict as edict

from icepy4d.thirdparty.SuperGlue.models.matching import Matching
from icepy4d.thirdparty.SuperGlue.models.superglue import SuperGlue
from icepy4d.thirdparty.SuperGlue.models.superpoint import SuperPoint
from icepy4d.thirdparty.SuperGlue.models.utils import make_matching_plot
from icepy4d.utils import AverageTimer

# SuperPoint Parameters
NMS_RADIUS = 3

# SuperGlue Parameters
SUPERGLUE_DESC_DIM = 256
SINKHORN_ITERATIONS = 10


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.0).float()[None, None].to(device)


class MatcherABC(ABC):
    def __init__(self, opt: dict = {}) -> None:
        self._opt = edict(opt)

    @abstractmethod
    def match(self):
        pass

    @abstractmethod
    def geometric_verification(self):
        pass


def check_dict_keys(dict: dict, keys: List[str]):
    missing_keys = [key for key in keys if key not in dict]
    if missing_keys:
        raise KeyError(
            f"Missing required keys: {', '.join(missing_keys)} Matcher option dictionary"
        )


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
    }


class MatcherBase(MatcherABC):
    def __init__(self, opt: dict = {}) -> None:
        """Base class for matchers"""
        if not isinstance(opt, dict):
            raise TypeError("opt must be a dictionary")
        self._opt = edict(opt)

        # self.features0 and self.features1 are dictionaries containing keypoints, descriptors and scores for each image and it is set by the method self.match()
        self._features0 = {}  # Keypoints, descriptors and scores for image 0
        self._features1 = {}  # Keypoints, descriptors and scores for image 1
        self._matches0 = None  # Index of the matches from image 0 to image 1
        self._mconf = None  # Confidence of the matches

        self._device = (
            "cuda"
            if torch.cuda.is_available() and not opt.get("force_cpu", "True")
            else "cpu"
        )
        logging.info(f"Running inference on device {self._device}")

    @property
    def features0(self):
        return self._features0

    @features0.setter
    def features0(self, value):
        self._features0 = value

    @property
    def features1(self):
        return self._features1

    @features1.setter
    def features1(self, value):
        self._features1 = value

    @property
    def matches0(self):
        return self._matches0

    @matches0.setter
    def matches0(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("matches0 must be a NumPy array")
        self._matches0 = value

        # Store index of the valid matches on image 0
        self._valid_idx_0 = self.matches0 > -1
        # Store index of the valid matches on image 1
        self._valid_idx_1 = self.matches0[self._valid_idx_0]

    @property
    def match_conf(self):
        return self._features0.get("scores")[self._valid_idx_0]

    @match_conf.setter
    def match_conf(self, value):
        self._match_conf = value

    @property
    def keypoints0(self):
        return self._features0.get("keypoints")[self._valid_idx_0]

    @property
    def keypoints1(self):
        return self._features1.get("keypoints")[self._valid_idx_1]

    @property
    def descriptors0(self):
        return self._features0.get("descriptors")[:, self._valid_idx_0]

    @property
    def descriptors1(self):
        return self._features1.get("descriptors")[:, self._valid_idx_1]

    @property
    def scores0(self):
        return self._features0.get("scores")[self._valid_idx_0]

    @property
    def scores1(self):
        return self._features1.get("scores")[self._valid_idx_1]

    @property
    def mkpts0(self):
        "Same as self.keypoints0 Kept for backward compatibility"
        return self.keypoints0

    @property
    def mkpts1(self):
        "Same as self.keypoints1 Kept for backward compatibility"
        return self.keypoints1

    def match(self):
        """
        Matches keypoints between two images. This method must be implemented by the child class.
        """
        return super().match()

    def geometric_verification(
        self,
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

        """
        assert (
            self.mkpts0 is not None and self.mkpts1 is not None
        ), "Matches not available."

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
                self._F, self._inlMask = pydegensac.findFundamentalMatrix(
                    self.mkpts0,
                    self.mkpts1,
                    px_th=threshold,
                    conf=confidence,
                    max_iters=max_iters,
                    laf_consistensy_coef=laf_consistensy_coef,
                    error_type=error_type,
                    symmetric_error_check=symmetric_error_check,
                    enable_degeneracy_check=enable_degeneracy_check,
                )
                logging.info(
                    f"Pydegensac found {self._inlMask.sum()} inliers ({self._inlMask.sum()*100/len(self.mkpts0):.2f}%)"
                )
            except Exception as err:
                # Fall back to MAGSAC++ if pydegensac fails
                logging.error(
                    f"{err}. Unable to perform geometric verification with Pydegensac. Trying using MAGSAC++ (OpenCV) instead."
                )
                fallback = True

        if method == "opencv" or fallback:
            try:
                self._F, inliers = cv2.findFundamentalMat(
                    self.mkpts0, self.mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000
                )
                self._inlMask = (inliers > 0).squeeze()
                logging.info(
                    f"MAGSAC++ found {self._inlMask.sum()} inliers ({self._inlMask.sum()*100/len(self.mkpts0):.2f}%)"
                )
            except Exception as err:
                logging.error(
                    f"{err}. Unable to perform geometric verification with MAGSAC++."
                )
                self._inlMask = np.ones(len(self.mkpts0), dtype=bool)

        self.mkpts0 = self.mkpts0[self._inlMask, :]
        self.mkpts1 = self.mkpts1[self._inlMask, :]
        if self.matches0 is not None:
            self.match_conf = self.match_conf[self._inlMask]
        if self.descriptors0 is not None:
            self.descriptors0 = self.descriptors0[:, self._inlMask]
        if self.descriptors1 is not None:
            self.descriptors1 = self.descriptors1[:, self._inlMask]
        if self.scores0 is not None:
            self.scores0 = self.scores0[self._inlMask]
        if self.scores1 is not None:
            self.scores1 = self.scores1[self._inlMask]

        return {0: self.mkpts0, 1: self.mkpts1}

    def viz_matches(
        self,
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
        assert self.mkpts0 is not None, "Matches not available."
        # image0 = np.uint8(tensor0.cpu().numpy() * 255),

        color = cm.jet(self.match_conf)
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(
                len(self.features0.keypoints), len(self.features1.keypoints)
            ),
            "Matches: {}".format(len(self.mkpts0)),
        ]

        # Display extra parameter info.
        k_thresh = self.opt["keypoint_threshold"]
        m_thresh = self.opt["match_threshold"]
        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
        ]

        make_matching_plot(
            self.image0,
            self.image1,
            self.mkpts0,
            self.mkpts1,
            self.mkpts0,
            self.mkpts1,
            color,
            text,
            path=path,
            show_keypoints=show_keypoints,
            fast_viz=fast_viz,
            opencv_display=opencv_display,
            opencv_title="Matches",
            small_text=small_text,
        )


class SuperGlueMatcher(MatcherBase):
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
        self.config = self._opt

    def match(
        self, image0: np.ndarray, image1: np.ndarray, store_images: bool = False
    ) -> dict:
        """Matches keypoints and descriptors in two images using the SuperGlue algorithm.

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

        if store_images:
            self._image0 = image0
            self._image1 = image1

        tensor0 = frame2tensor(image0, self._device)
        tensor1 = frame2tensor(image1, self._device)

        self.timer = AverageTimer()
        matching = Matching(self.config).eval().to(self._device)
        with torch.inference_mode():
            self._pred_tensor = matching({"image0": tensor0, "image1": tensor1})
        pred = {k: v[0].cpu().numpy() for k, v in self._pred_tensor.items()}

        self.features0 = {
            "keypoints": pred["keypoints0"],
            "descriptors": pred["descriptors0"],
            "scores": pred["scores0"],
        }
        self.features1 = {
            "keypoints": pred["keypoints1"],
            "descriptors": pred["descriptors1"],
            "scores": pred["scores1"],
        }
        self.matches0 = pred["matches0"]
        self.match_conf = pred["matching_scores0"][self.matches0 > -1]

        # Extraction of matched keypoints, descriptors and scores is not needed anymore because is implemeted automatically in the base class.
        # valid_idx_0 = self.matches0 > -1
        # valid_idx_1 = self.matches0[valid_idx_0]
        # self.mkpts0 = self.features0.keypoints[valid_idx_0]
        # self.mkpts1 = self.features1.keypoints[valid_idx_1]
        # self.descriptors0 = self.features0.descriptors[:, valid_idx_0]
        # self.descriptors1 = self.features1.descriptors[:, valid_idx_1]
        # self.scores0 = self.features0.scores[valid_idx_0]
        # self.scores1 = self.features1.scores[valid_idx_1]
        # self.match_conf = pred["matching_scores0"][valid_idx_0]

        self.timer.update("matching")
        self.timer.print("SuperGlue matching")

        return {0: self.keypoints0, 1: self.keypoints1}


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
    mkpts = matcher.match(img0, img1)
    mkpts = matcher.geometric_verification(
        method="ransac",
        threshold=1,
        confidence=0.99,
        symmetric_error_check=False,
    )
    matcher.viz_matches(assset_path / "test.png")

    os.remove(assset_path / "test.png")

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
