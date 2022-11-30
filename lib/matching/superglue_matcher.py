"""SuperGlue matcher implementation

The network was proposed in 'SuperGlue: Learning Feature Matching with Graph Neural Networks' and is implemented by
wrapping over author's source-code.

Note: the pretrained model only supports SuperPoint detections currently.

References:
- http://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
- https://github.com/magicleap/SuperGluePretrainedNetwork

Authors:
"""

import numpy as np
import matplotlib.cm as cm
import torch

from easydict import EasyDict as edict
from pathlib import Path
from typing import Tuple, Union

from lib.base_classes.features import Features
from lib.base_classes.images import Imageds
from lib.read_config import parse_yaml_cfg

from thirdparty.SuperGluePretrainedNetwork.superpoint import SuperPoint
from thirdparty.SuperGluePretrainedNetwork.superglue import SuperGlue
from thirdparty.SuperGluePretrainedNetwork.utils import read_image, frame2tensor

# make_matching_plot, AverageTimer, read_image, vizTileRes

# from lib.sg.matching import Matching

from lib.tiles import generateTiles

torch.set_grad_enabled(False)

# SuperPoint constants
# MODEL_WEIGHTS_PATH = Path(
#     "thirdparty/SuperGluePretrainedNetwork/weights/superpoint_v1.pth")
NMS_RADIUS = 3


SUPERGLUE_DESC_DIM = 256

# SuperGlue Hyperparameters
DEFAULT_NUM_SINKHORN_ITERATIONS = 20


class SuperPoint(torch.nn.Module):
    """Image Matching Frontend (SuperPoint + SuperGlue)"""

    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get("superpoint", {}))

    def forward(self, data):
        """Run SuperPoint
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """

        # Extract SuperPoint (keypoints, scores, descriptors)
        pred = self.superpoint({"image": data["image0"]})
        pred = {**pred, **{k + "0": v for k, v in pred0.items()}}

        return pred


class SuperPoint_detector_descriptor:
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
        image = read_image(im_path, device)
        image_tensor = frame2tensor(image)

        # Compute features.
        with torch.no_grad():
            model_results = model({"image": image_tensor})
        torch.cuda.empty_cache()

        keypoints = model_results["keypoints"][0].detach().cpu().numpy()
        scores = model_results["scores"][0].detach().cpu().numpy()
        # keypoints = Keypoints(coordinates, scales=None, responses=scores)
        descriptors = model_results["descriptors"][0].detach().cpu().numpy()

        features = Features
        features.append_features(
            {
                "kpts": keypoints,
                "descr": descriptors,
                "score": scores,
            }
        )

        return features


# class SuperGlue():
#     """Implements the SuperGlue matcher -- a pretrained graph neural network using attention."""

#     def __init__(self,
#                  #  image0: np.ndarray,
#                  #  image1: np.ndarray,
#                  use_cuda: bool = True,
#                  use_outdoor_model: bool = True,
#                  #  max_keypoints: int = 2048,
#                  #  keypoint_threshold: float = 0.0001,
#                  match_threshold: float = 0.15,
#                  config: edict = None,
#                  ) -> None:
#         """Initialize the configuration and the parameters."""

#         self._config = {
#             'superpoint': {
#                 'nms_radius': NMS_RADIUS,
#                 'keypoint_threshold': keypoint_threshold,
#                 'max_keypoints': max_keypoints
#             },
#             'superglue': {
#                 'weights': "outdoor" if use_outdoor_model else "indoor",
#                 'sinkhorn_iterations': DEFAULT_NUM_SINKHORN_ITERATIONS,
#                 'match_threshold': match_threshold,
#             }
#         }

#         self._use_cuda = use_cuda and torch.cuda.is_available()

#     def detect_and_match(self,
#                          max_keypoints: int = 5000,
#                          use_cuda: bool = True,
#                          weights_path: Path = MODEL_WEIGHTS_PATH,
#                          use_cudaopt: edict):
#         """ Run Superpoint for finding features and match them with SuperGlue

#         Args:
#             max_keypoints: max keypoints to detect in an image.
#             use_cuda (optional): flag controlling the use of GPUs via CUDA. Defaults to True.
#             weights_path (optional): Path to the model weights. Defaults to MODEL_WEIGHT_PATH.
#         """

#         device = torch.device("cuda" if self._use_cuda else "cpu")
#         model = SuperPoint(self._config).to(device)
#         model.eval()

#         # Compute features.
#         image_tensor = torch.from_numpy(
#             np.expand_dims(image_utils.rgb_to_gray_cv(
#                 image).value_array.astype(np.float32) / 255.0, (0, 1))
#         ).to(device)
#         with torch.no_grad():
#             model_results = model({"image": image_tensor})
#         torch.cuda.empty_cache()

#         # Unpack results.
#         coordinates = model_results["keypoints"][0].detach().cpu().numpy()
#         scores = model_results["scores"][0].detach().cpu().numpy()
#         keypoints = Keypoints(coordinates, scales=None, responses=scores)
#         descriptors = model_results["descriptors"][0].detach().cpu().numpy().T

#         # Filter features.
#         if image.mask is not None:
#             keypoints, valid_idxs = keypoints.filter_by_mask(image.mask)
#             descriptors = descriptors[valid_idxs]
#         keypoints, selection_idxs = keypoints.get_top_k(self.max_keypoints)
#         descriptors = descriptors[selection_idxs]

#         return keypoints, descriptors

#         print('')


if __name__ == "__main__":

    import cv2
    from base_classes.classes_old import Imageds, Features

    cfg_file = "config/config_base.yaml"
    cfg = parse_yaml_cfg(cfg_file)

    cams = cfg.paths.cam_names

    # Create Image Datastore objects
    images = dict.fromkeys(cams)
    for cam in cams:
        images[cam] = Imageds(cfg.paths.imdir / cam)

    superpoint_detector = SuperPoint_detector_descriptor(cfg.matching.max_keypoints)

    features = superpoint_detector.detect_and_describe(
        images[cams[0]].get_image_path(0)
    )
