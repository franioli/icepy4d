import cv2
import numpy as np


def homography_warping(
    cam_0: np.ndarray,
    cam_1: np.ndarray,
    image: np.ndarray,
    out_path: str = None,
) -> np.ndarray:
    print("Performing homography warping based on extrinsics matrix...")

    # Create deepcopies to not modify original data
    cam_0_ = deepcopy(cam_0)
    cam_1_ = deepcopy(cam_1)

    T = np.linalg.inv(cam_0_.pose)
    cam_0_.update_extrinsics(cam_0_.pose_to_extrinsics(T @ cam_0_.pose))
    cam_1_.update_extrinsics(cam_1_.pose_to_extrinsics(T @ cam_1_.pose))

    R = cam_1_.R
    K = cam_1_.K
    H = (cam_0_.K @ R) @ np.linalg.inv(K)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    w, h = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (h, w))
    if out_path is not None:
        cv2.imwrite(out_path, warped_image)
        print(f"Warped image {Path(out_path).stem} exported correctely")

    return cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
