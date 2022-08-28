"""SuperGlue matcher implementation

The network was proposed in "SuperGlue: Learning Feature Matching with Graph Neural Networks" and is implemented by
wrapping over author"s source-code.

Note: the pretrained model only supports SuperPoint detections currently.

References:
- http://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
- https://github.com/magicleap/SuperGluePretrainedNetwork

Authors: Francesco Ioli
"""

import numpy as np
import matplotlib.cm as cm
import torch

from easydict import EasyDict as edict
from pathlib import Path
from typing import Tuple, Union

from lib.classes import Imageds, Features
from lib.config import parse_yaml_cfg

from thirdparty.SuperGluePretrainedNetwork.matching import Matching
from thirdparty.SuperGluePretrainedNetwork.superpoint import SuperPoint
from thirdparty.SuperGluePretrainedNetwork.superglue import SuperGlue
from thirdparty.SuperGluePretrainedNetwork.utils import (
    read_image, frame2tensor,
    make_matching_plot, AverageTimer,
)

torch.set_grad_enabled(False)

# SuperPoint constants
NMS_RADIUS = 3


SUPERGLUE_DESC_DIM = 256

# SuperGlue Hyperparameters
DEFAULT_NUM_SINKHORN_ITERATIONS = 20


class SuperGlue():
    """Implements the SuperGlue matcher -- a pretrained graph neural network using attention.    
    """
    # @TODO: change function in order to accept either the image paths or the images directly (in order to work with tiles)
    
    def __init__(self,           
                image_pair_paths: list,
                image_pair: list = None,                 
                use_cuda: bool = True,
                max_keypoints: int = 2048,
                keypoint_threshold: float = 0.0001,
                use_outdoor_model: bool = True,
                match_threshold: float = 0.15,
                resize: list = [-1],
                output_dir: Path = Path("res"),
                opt: edict = None,
                ) -> None:
        """Initialize the configuration and the parameters.
            Args:
                - image_pair
                - use_cuda (optional): flag controlling the use of GPUs via CUDA. Defaults to True.
                - max_keypoints: max keypoints to detect in an image.
        """

        self._config = {
            "superpoint": {
                "nms_radius": NMS_RADIUS,
                "keypoint_threshold": keypoint_threshold,
                "max_keypoints": max_keypoints
            },
            "superglue": {
                "weights": "outdoor" if use_outdoor_model else "indoor",
                "sinkhorn_iterations": DEFAULT_NUM_SINKHORN_ITERATIONS,
                "match_threshold": match_threshold,
            }
        }

        # Check whether using GPU with CUDA or CPU
        if use_cuda and torch.cuda.is_available():
            self.device =  torch.device("cuda")
        else:
            self.device =  torch.device("cpu")

        # Check if images are given or if they must be read from image_pair_path
        if image_pair is not None:
            self.im_pair = image_pair
        else:
            self.im_pair = None

        # Resize input images
        self.resize = resize
       
        # Prepare paths and folders
        self.paths = (Path(image_pair_paths[0]), Path(image_pair_paths[1]))
        self.stems = (self.paths[0].stem, self.paths[1].stem)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Will write matches to directory {output_dir}")
        
        self.matches_path = self.output_dir / \
            f"{self.stems[0]}_{self.stems[0]}_matches.npz"
        self.viz_path = self.output_dir / \
            f"{self.stems[0]}_{self.stems[0]}_matches.png"

        # Set defautl additional options and check if value is given in opt dict
        self.do_viz = False
        self.show_keypoints = False
        if opt is not None:
            if 'do_viz' in opt.keys():
                self.do_viz = opt.do_viz
            if 'show_keypoints' in opt.keys():
                self.show_keypoints = opt.show_keypoints

        # Start timer
        self.timer = AverageTimer(newline=True)


    def read_image_pair(self) -> None:
        
        # Assert resize option
        if len(self.resize) == 2:
            print("Will resize to {}x{} (WxH)".format(
                self.resize[0], self.resize[1]))
        elif len(self.resize) == 1 and self.resize[0] > 0:
            print("Will resize max dimension to {}".format(self.resize[0]))
        elif len(self.resize) == 1:
            print("Will not resize images")
        else:
            raise ValueError("Invalid image resize parameter")
    
        # Read images
        image0, _, scales0 = read_image(
            self.paths[0], self.device, resize = self.resize
            ) 
        image1, _, scales1 = read_image(
            self.paths[1], self.device, resize = self.resize
            )       
        if image0 is None or image1 is None:
            raise ValueError('Problem reading the images')
                       
        self.timer.update("load_image")
        
        self.im_pair = [image0, image1]
                
        
    def detect_and_match(self) -> dict: 
        """ Run Superpoint for finding features and match them with SuperGlue
        """

        # Initializing Matching object
        matching = Matching(self._config).eval().to(self.device)
        
        # Check if images are given as input
        if self.im_pair is None:
            self.read_image_pair()
 
        image0_tensor = frame2tensor(self.im_pair[0], self.device)
        image1_tensor = frame2tensor(self.im_pair[1], self.device)
    
        # Run prediction
        pred = matching({'image0': image0_tensor, 'image1': image1_tensor})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}   
        self.timer.update('matcher')

        # Write the matches to disk.
        np.savez(str(self.matches_path), **pred)

        # Free cuda memory and return variables
        torch.cuda.empty_cache()
        
        if self.do_viz:
            self.viz_result(pred)  
        
        self.timer.print('Finished pair')

        return pred

    def viz_result(self, pred) -> None:
        # Visualize the matches.
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches0 = pred['matches0']
        valid = matches0 > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches0[valid]]                    
        mconf = pred['matching_scores0'][valid]

        color = cm.jet(mconf)
        text = [
            'SuperGlue',
            f'Keypoints: {len(kpts0)}:{len(kpts1)}',
            f'Matches: {len(mkpts0)}',
        ]

        # Display extra parameter info.
        k_thresh = self._config['superpoint']['keypoint_threshold']
        m_thresh = self._config['superglue']['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            f'Image Pair: {self.stems[0]}:{self.stems[1]}',
        ]

        make_matching_plot(
            self.im_pair[0], self.im_pair[0], 
            kpts0, kpts1,
            mkpts0, mkpts1,
            color, text, self.viz_path, self.show_keypoints,
            True, False, 'Matches', small_text)

        self.timer.update('viz_match')


if __name__ == "__main__":

    import cv2
    from lib.classes import Imageds, Features

    cfg_file = "config/config_base.yaml"
    cfg = parse_yaml_cfg(cfg_file)

    cams = cfg.paths.cam_names

    # Create Image Datastore objects
    images = dict.fromkeys(cams)
    for cam in cams:
        images[cam] = Imageds(cfg.paths.imdir / cam)

    pair = ['IMG_0520.jpg', 'IMG_2131.jpg']
    
    matcher = SuperGlue(pair, opt=cfg.matching)
    pred = matcher.detect_and_match()
    
    