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


from ..base_classes.features import Features
from ..base_classes.images import ImageDS
from ..utils.initialization import parse_yaml_cfg
from ..tiles import generateTiles

from ..thirdparty.SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from ..thirdparty.SuperGluePretrainedNetwork.models.superglue import SuperGlue
from ..thirdparty.SuperGluePretrainedNetwork.models.utils import (
    read_image,
    frame2tensor,
)


torch.set_grad_enabled(False)

# SuperPoint constants
# MODEL_WEIGHTS_PATH = Path(
#     "thirdparty/SuperGluePretrainedNetwork/weights/superpoint_v1.pth")
NMS_RADIUS = 3


SUPERGLUE_DESC_DIM = 256

# SuperGlue Hyperparameters
DEFAULT_NUM_SINKHORN_ITERATIONS = 20


# class SuperPoint_(torch.nn.Module):
#     """Image Matching Frontend (SuperPoint + SuperGlue)"""

#     def __init__(self, config={}):
#         super().__init__()
#         self.superpoint = SuperPoint(config.get("superpoint", {}))

#     def forward(self, data):
#         """Run SuperPoint
#         Args:
#           data: dictionary with minimal keys: ['image']
#         """

#         # Extract SuperPoint (keypoints, scores, descriptors)
#         pred = self.superpoint({"image": data["image"]})
#         pred = {**pred, **{k + "0": v for k, v in pred.items()}}

#         return pred


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


# class SuperGlue_matcher():
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
