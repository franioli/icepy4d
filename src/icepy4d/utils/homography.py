import logging
from copy import deepcopy
from pathlib import Path
from typing import Union

import cv2
import numpy as np


def homography_warping(
    cam_0: np.ndarray,
    cam_1: np.ndarray,
    image: np.ndarray,
    undistort: bool = False,
    out_path: Union[str, Path] = None,
) -> np.ndarray:
    logging.info(f"Performing homography warping based on extrinsics matrix...")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create deepcopies to not modify original data
    cam_0_ = deepcopy(cam_0)
    cam_1_ = deepcopy(cam_1)

    # Convert colors to OpenCV format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if undistort:
        image = cv2.undistort(image, cam_1_.K, cam_1_.dist, None, cam_1_.K)
        logging.info(f"Distortion corrected.")

    T = np.linalg.inv(cam_0_.pose)
    cam_0_.update_extrinsics(cam_0_.pose_to_extrinsics(T @ cam_0_.pose))
    cam_1_.update_extrinsics(cam_1_.pose_to_extrinsics(T @ cam_1_.pose))

    R = cam_1_.R
    K = cam_1_.K
    H = (cam_0_.K @ R) @ np.linalg.inv(K)

    h, w = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (w, h))

    if out_path is not None:
        cv2.imwrite(str(out_path), warped_image)
        logging.info(
            f"Warped image {Path(out_path).stem} exported correctely to {out_path}"
        )

    return cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
