"""SuperGlue tracker implementation
The network was proposed in 'SuperGlue: Learning Feature Matching with Graph Neural Networks' and is implemented by wrapping over author's source-code.
Note: the pretrained model only supports SuperPoint detections currently.
References:
- http://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
- https://github.com/magicleap/SuperGluePretrainedNetwork
"""

import numpy as np
import matplotlib.cm as cm
import torch
import logging
import cv2
import importlib

from easydict import EasyDict as edict
from pathlib import Path
from typing import Tuple, Union


from ..classes.features import Features
from ..classes.images import ImageDS
from ..utils.initialization import parse_yaml_cfg
from ..tiles import generateTiles


from icepy4d.matching.utils import read_image, frame2tensor
from icepy4d.utils import AverageTimer


from icepy4d.thirdparty.SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from icepy4d.thirdparty.SuperGluePretrainedNetwork.models.superglue import SuperGlue
from icepy4d.thirdparty.SuperGluePretrainedNetwork.models.matching import Matching
from icepy4d.thirdparty.SuperGluePretrainedNetwork.models.utils import (
    make_matching_plot,
)


# SuperPoint Parameters
NMS_RADIUS = 3

# SuperGlue Parameters
SUPERGLUE_DESC_DIM = 256
SINKHORN_ITERATIONS = 20


class SuperGlueTracker:
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
        if not isinstance(opt, dict):
            raise TypeError("opt must be a dictionary")
        required_keys = [
            "weights",
            "keypoint_threshold",
            "max_keypoints",
            "match_threshold",
            "force_cpu",
        ]
        missing_keys = [key for key in required_keys if key not in opt]
        if missing_keys:
            raise KeyError(
                f"Missing required keys: {', '.join(missing_keys)} in SuperGlue Tracker option dictionary"
            )
        self.opt = edict(opt)

        self.features0 = None
        self.features1 = None
        self.matches0 = None
        self.matches1 = None
        self.match_conf = None
        self.mkpts0 = None
        self.mkpts1 = None

        self.device = (
            "cuda" if torch.cuda.is_available() and not self.opt.force_cpu else "cpu"
        )
        logging.info(f"Running inference on device {self.device}")

        self.config = {
            "superpoint": {
                "nms_radius": NMS_RADIUS,
                "keypoint_threshold": self.opt.keypoint_threshold,
                "max_keypoints": self.opt.max_keypoints,
            },
            "superglue": {
                "weights": self.opt.weights,
                "sinkhorn_iterations": SINKHORN_ITERATIONS,
                "match_threshold": self.opt.match_threshold,
            },
        }

    def match(self, image0: np.ndarray, image1: np.ndarray) -> dict:
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

        self.image0 = image0
        self.image1 = image1

        tensor0 = frame2tensor(image0, self.device)
        tensor1 = frame2tensor(image1, self.device)

        self.timer = AverageTimer()
        matching = Matching(self.config).eval().to(self.device)
        self.pred = matching({"image0": tensor0, "image1": tensor1})
        self.timer.update("matching")
        unpack = {k: v[0].cpu().numpy() for k, v in self.pred.items()}

        self.features0 = edict(
            {
                "keypoints": unpack["keypoints0"],
                "descriptors": unpack["descriptors0"],
                "scores": unpack["scores0"],
            }
        )
        self.features1 = edict(
            {
                "keypoints": unpack["keypoints1"],
                "descriptors": unpack["descriptors1"],
                "scores": unpack["scores1"],
            }
        )
        self.matches0 = unpack["matches0"]
        self.matches1 = unpack["matches1"]

        valid_idx_0 = self.matches0 > -1
        valid_idx_1 = self.matches0[valid_idx_0]
        self.mkpts0 = self.features0.keypoints[valid_idx_0]
        self.mkpts1 = self.features1.keypoints[valid_idx_1]
        self.descriptors0 = self.features0.descriptors[:, valid_idx_0]
        self.descriptors1 = self.features1.descriptors[:, valid_idx_1]
        self.scores0 = self.features0.scores[valid_idx_0]
        self.scores1 = self.features1.scores[valid_idx_1]
        self.match_conf = unpack["matching_scores0"][valid_idx_0]

        self.timer.print("SuperGlue matching")

        return {0: self.mkpts0, 1: self.mkpts1}

    def geometric_verification(
        self,
        threshold: float = 1,
        confidence: float = 0.9999,
        max_iters: int = 10000,
        laf_consistensy_coef: float = -1.0,
        error_type: str = "sampson",
        symmetric_error_check: bool = True,
        enable_degeneracy_check: bool = True,
    ) -> np.ndarray:
        """
        Computes the fundamental matrix and inliers between the two images using geometric verification.

        Args:
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
        assert self.mkpts0 is not None, "Matches not available."

        try:
            pydegensac = importlib.import_module("pydegensac")
            self.F, self.inlMask = pydegensac.findFundamentalMatrix(
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
                f"Pydegensac found {self.inlMask.sum()} inliers ({self.inlMask.sum()*100/len(self.mkpts0):.2f}%)"
            )
        except:
            logging.error(
                "Pydegensac not available. Using MAGSAC++ (OpenCV) for geometric verification."
            )
            self.F, inliers = cv2.findFundamentalMat(
                self.mkpts0, self.mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000
            )
            self.inlMask = inliers > 0
            logging.info(
                f"MAGSAC++ found {self.inlMask.sum()} inliers ({self.inlMask.sum()*100/len(self.mkpts0):.2f}%)"
            )

        self.mkpts0 = self.mkpts0[self.inlMask]
        self.mkpts1 = self.mkpts1[self.inlMask]

        return {0: self.mkpts0, 1: self.mkpts1, "inliers": self.inlMask}

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


class SuperPoint_features:
    def __init__(
        self,
        max_keypoints: int = 2048,
        keypoint_threshold: float = 0.0001,
        use_cuda: bool = True,
        #  weights_path: Path = MODEL_WEIGHTS_PATH,
    ) -> None:
        """Configures the object.
        Args:
            max_keypoints: max keypoints to detect in an image.
            keypoint_threshold: threshold for keypoints detection
            use_cuda (optional): flag controlling the use of GPUs via CUDA. Defaults to True.
            # weights_path (optional): Path to the model weights. Defaults to MODEL_WEIGHT_PATH.
        """
        self._use_cuda = use_cuda and torch.cuda.is_available()
        self._config = {
            "superpoint": {
                "nms_radius": NMS_RADIUS,
                "keypoint_threshold": keypoint_threshold,
                "max_keypoints": max_keypoints,
            },
        }

    def detect_and_describe(self, im_path: Path):
        """Jointly generate keypoint detections and their associated descriptors from a single image."""
        # TODO(ayushbaid): fix inference issue #110
        device = torch.device("cuda" if self._use_cuda else "cpu")
        model = SuperPoint(self._config).eval().to(device)

        # Read image and transform to tensor
        image, image_tensor, _ = read_image(im_path, device, [2400], 0, True)

        # Compute features.
        with torch.no_grad():
            model_results = model({"image": image_tensor})
        torch.cuda.empty_cache()

        return model_results

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

    cfg_file = "config/config_base.yaml"
    cfg = parse_yaml_cfg(cfg_file)

    cams = cfg.paths.camera_names

    # Create Image Datastore objects
    images = dict.fromkeys(cams)
    for cam in cams:
        images[cam] = ImageDS(cfg.paths.image_dir / cam)

    img0 = images[cams[0]].get_image_path(0)
    img1 = images[cams[1]].get_image_path(0)

    superpoint_detector = SuperPoint_features(cfg.matching.max_keypoints)
    features0 = superpoint_detector.detect_and_describe(img0)
    features1 = superpoint_detector.detect_and_describe(img1)

    device = torch.device("cuda")
    matching = Matching().eval().to(device)

    _, tens0, _ = read_image(img0, device, [2400], 0, True)
    _, tens1, _ = read_image(img1, device, [2400], 0, True)

    data = {
        "image0": tens0,
        "image1": tens1,
    }
    data = {**data, **{k + "0": v for k, v in features0.items()}}
    data = {**data, **{k + "1": v for k, v in features1.items()}}

    pred = matching(data)
