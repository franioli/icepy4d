import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from icepy4d.classes.features import Features

logger = logging.getLogger(__name__)


# def geometric_verification(
#     features: icepy4d_classes.FeaturesDict,
#     threshold: float = 1,
#     confidence: float = 0.9999,
#     max_iters: int = 10000,
#     laf_consistensy_coef: float = -1.0,
#     error_type: str = "sampson",
#     symmetric_error_check: bool = True,
#     enable_degeneracy_check: bool = True,
# ):
#     """
#     NOTE: this function is deprecated. Use geometric_verification
#     from icepy4d.matching.geometric_verification instead.

#     Performs geometric verification of matches on full images using
#     either Pydegensac or MAGSAC++.

#     Args:
#         features (icepy4d_classes.FeaturesDict): A dictionary containing
#         extracted features for each camera view.
#             threshold (float): Pixel error threshold for considering a
#             correspondence an inlier.
#             confidence (float): The required confidence level in the results.
#             max_iters (int): The maximum number of iterations for estimating
#             the fundamental matrix.
#             laf_consistensy_coef (float): The weight given to Local Affine
#             Frame (LAF) consistency term for pydegensac.
#             error_type (str): The error function used for computing the
#             residuals in the RANSAC loop.
#             symmetric_error_check (bool): If True, performs an additional check
#             on the residuals in the opposite direction.
#             enable_degeneracy_check (bool): If True, enables the check for
#             degeneracy using SVD.

#     Returns:
#         None

#     Raises:
#         ImportError: If Pydegensac library is not available.

#     Note:
#         The function modifies directly the input Features object by removing
#         the outliers after geoemtric verification.

#     """

#     logger.warning(
#         "This function is deprecated. Use geometric_verification from icepy4. Use matching.geometric_verification instead."
#     )

#     logger.info("Geometric verification of matches on full images...")
#     cams = list(features.keys())
#     try:
#         pydegensac = importlib.import_module("pydegensac")
#         use_pydegensac = True
#     except ImportError:
#         logger.error(
#             "Pydegensac not available. Using MAGSAC++ (OpenCV) for geometric verification."
#         )
#         use_pydegensac = False
#     if use_pydegensac:
#         F, inlMask = pydegensac.findFundamentalMatrix(
#             features[cams[0]].kpts_to_numpy(),
#             features[cams[1]].kpts_to_numpy(),
#             px_th=threshold,
#             conf=confidence,
#             max_iters=max_iters,
#             laf_consistensy_coef=laf_consistensy_coef,
#             error_type=error_type,
#             symmetric_error_check=symmetric_error_check,
#             enable_degeneracy_check=enable_degeneracy_check,
#         )
#         logger.info(
#             f"Pydegensac found {inlMask.sum()} inliers ({inlMask.sum()*100/len(features[cams[0]]):.2f}%)"
#         )
#     else:
#         F, inliers = cv2.findFundamentalMat(
#             features[cams[0]].kpts_to_numpy(),
#             features[cams[1]].kpts_to_numpy(),
#             cv2.USAC_MAGSAC,
#             0.5,
#             0.999,
#             100000,
#         )
#         inlMask = inliers > 0
#         logger.info(
#             f"MAGSAC++ found {inlMask.sum()} inliers ({inlMask.sum()*100/len(features[cams[0]].kpts_to_numpy()):.2f}%)"
#         )

#     features[cams[0]].filter_feature_by_mask(inlMask)
#     features[cams[1]].filter_feature_by_mask(inlMask)


def process_resize(w, h, resize):
    assert len(resize) > 0 and len(resize) <= 2
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # # Issue warning if resolution is too small or too large.
    # if max(w_new, h_new) < 160:
    #     logger.warning("Warning: input resolution is very small, results may vary")
    # elif max(w_new, h_new) > 2000:
    #     logger.warning("Warning: input resolution is very large, results may vary")

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.0).float()[None, None].to(device)


def read_image(path, device, resize, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype("float32"), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype("float32")

    tensor = frame2tensor(image, device)
    return image, tensor, scales


# -- File I/O --#
def load_matches_from_disk(dir: Path) -> Features:
    """
    load_matches_from_disk Load features from pickle file

    Args:
        dir (Union[str, Path]): path of the folder containing the pickle file

    Raises:
        FileNotFoundError: No pickle file found or multiple pickle file found.

    Returns:
        Features: Loaded features
    """
    try:
        fname = list(dir.glob("*.pickle"))
        if len(fname) < 1:
            msg = f"No pickle file found in the epoch directory {dir}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        if len(fname) > 1:
            msg = f"More than one pickle file is present in the epoch directory {dir}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        with open(fname[0], "rb") as f:
            try:
                loaded_features = pickle.load(f)
                logger.info(f"Loaded features from {fname[0]}")
                return loaded_features
            except:
                msg = f"Invalid pickle file in epoch directory {dir}"
                logger.error(msg)
                raise FileNotFoundError(msg)
    except FileNotFoundError as err:
        logger.exception(err)


def retrieve_matches_from_npz(opt: dict):
    """
    retrieve_matches Retrieve matches saved by SuperGlue in a npz archive and save results in text files

    Args:
        opt (dict): dict with the following keys:
            - npz (str): path to the npz file
            - output_dir (str): output direcotry path
    Raises:
        IOError: unable to read npz file
    """

    npz_path = Path(opt.npz)
    output_path = Path(opt.output_dir)

    npz_path = Path(npz_path)
    if npz_path.exists():
        try:
            npz = np.load(npz_path)
        except:
            raise IOError(f"Cannot load matches .npz file {npz_path.name}")
    if "npz" in locals():
        print(f"Data in {npz_path.name} loaded successfully")
    else:
        print("No data loaded")

    output_path = Path(output_path)
    if not output_path.is_dir():
        output_path.mkdir()

    res = {k: npz[k] for k in npz.keys()}
    for k in npz.keys():
        np.savetxt(
            output_path.as_posix() + "/" + k + ".txt",
            res[k],
            fmt="%.2f",
            delimiter=",",
            newline="\n",
        )
    print("Data saved successfully")
