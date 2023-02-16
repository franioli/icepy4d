from easydict import EasyDict as edict
import importlib
import logging
import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import Tuple

import icepy.classes as icepy_classes
import icepy.visualization as icepy_viz
from icepy.classes.images import read_image
from icepy.matching.superglue_matcher import SuperGlueMatcher

from scipy.spatial import KDTree
from scipy.cluster.vq import kmeans, vq, whiten
from scipy.spatial.distance import pdist


def average_distance(points: np.ndarray):
    """Calculate the pairwise distances between all points and Return the mean of the distances"""
    return pdist(points).mean()


def find_centroids_kmeans(
    data: np.ndarray, n_cluster: int, viz_clusters: bool = False
) -> Tuple[np.ndarray]:
    data_withen = whiten(data)
    centroids, _ = kmeans(data_withen, n_cluster)
    classes, _ = vq(data_withen, centroids)
    centroids = np.array(
        [np.mean(data[classes == i, :], axis=0) for i in range(n_cluster)]
    )
    if viz_clusters:
        fig, ax = plt.subplots()
        ax.scatter(
            data[:, 0],
            data[:, 1],
            c=list(classes),
        )
        ax.scatter(centroids[:, 0], centroids[:, 1], c="r")

    return (centroids, classes)


# epoch = 184
# n_tiles = 8
# n_dist = 2


def match_by_preselection(
    images: icepy_classes.ImagesDict,
    feats: icepy_classes.FeaturesDictEpoch,
    epoch: int,
    cfg: edict,
    n_tiles: int = 8,
    n_dist: int = 2,
    viz_results: bool = True,
) -> icepy_classes.FeaturesDictEpoch:

    cams = cfg.paths.camera_names
    logging.info("Running matching on downscaled images")
    image0 = read_image(images[cams[0]].get_image_path(epoch), resize=[3000, 2000])[0]
    image1 = read_image(images[cams[1]].get_image_path(epoch), resize=[3000, 2000])[0]
    matcher = SuperGlueMatcher(cfg.matching)
    match_presel = matcher.match(image0, image1)
    match_presel = matcher.geometric_verification(
        threshold=20, confidence=0.9, symmetric_error_check=False
    )
    if viz_results:
        matcher.viz_matches(path="test_out/test_downscale.png", fast_viz=True)

    match_presel = {cams[i]: match_presel[i] * 2 for i, _ in enumerate(cams)}
    centroids, classes = find_centroids_kmeans(match_presel[cams[0]], n_tiles)

    logging.info("Running matching on full resolutions patches")
    feats = {cam: icepy_classes.Features() for cam in cams}
    for i, center in enumerate(centroids):
        logging.info(f"Processing tile {i}")
        win_size = int(
            n_dist * average_distance(match_presel[cams[0]][classes == i, :])
        )
        kdtree = KDTree(match_presel[cams[0]])
        dist, res = kdtree.query(center, 1)

        patches_lim = {
            cam: [
                int(match_presel[cam][res][0] - win_size),
                int(match_presel[cam][res][1] - win_size),
                int(match_presel[cam][res][0] + win_size),
                int(match_presel[cam][res][1] + win_size),
            ]
            for cam in cams
        }
        for cam in cams:
            if patches_lim[cam][0] < 0:
                patches_lim[cam][0] = 0
            if patches_lim[cam][1] < 0:
                patches_lim[cam][1] = 0
            if patches_lim[cam][2] > images[cam].read_image(epoch).width:
                patches_lim[cam][2] = images[cam].read_image(epoch).width
            if patches_lim[cam][3] > images[cam].read_image(epoch).height:
                patches_lim[cam][3] = images[cam].read_image(epoch).width
        patches = {
            cam: images[cam].read_image(epoch).extract_patch(patches_lim[cam])
            for cam in cams
        }
        # fig, axes = plt.subplots(1,2)
        # for i, ax in enumerate(axes):
        #     ax.imshow(patches[cams[i]])

        matcher = SuperGlueMatcher(cfg.matching)
        mkpts = matcher.match(patches[cams[0]], patches[cams[1]])
        mkpts = matcher.geometric_verification(
            threshold=10, confidence=0.99, symmetric_error_check=False
        )
        if viz_results:
            matcher.viz_matches(path=f"test_out/test_patch_{i}.png", fast_viz=True)

        mkpts = {
            cams[0]: matcher.mkpts0
            + np.array(patches_lim[cams[0]][0:2]).astype(np.int32),
            cams[1]: matcher.mkpts1
            + np.array(patches_lim[cams[1]][0:2]).astype(np.int32),
        }
        for cam in cams:
            feats[cam].append_features_from_numpy(mkpts[cam][:, 0], mkpts[cam][:, 1])

    logging.info("Geometric verification of matches on full images...")

    try:
        pydegensac = importlib.import_module("pydegensac")
        F, inlMask = pydegensac.findFundamentalMatrix(
            feats[cams[0]].kpts_to_numpy(),
            feats[cams[1]].kpts_to_numpy(),
            px_th=cfg.other.pydegensac_threshold,
            conf=cfg.other.pydegensac_confidence,
            max_iters=10000,
            laf_consistensy_coef=-1.0,
            error_type="sampson",
            symmetric_error_check=True,
            enable_degeneracy_check=True,
        )
        logging.info(
            f"Pydegensac found {inlMask.sum()} inliers ({inlMask.sum()*100/len(feats[cams[0]]):.2f}%)"
        )
    except:
        logging.error(
            "Pydegensac not available. Using MAGSAC++ (OpenCV) for geometric verification."
        )
        F, inliers = cv2.findFundamentalMat(
            feats[cams[0]].kpts_to_numpy(),
            feats[cams[1]].kpts_to_numpy(),
            cv2.USAC_MAGSAC,
            0.5,
            0.999,
            100000,
        )
        inlMask = inliers > 0
        logging.info(
            f"MAGSAC++ found {inlMask.sum()} inliers ({inlMask.sum()*100/len(feats[cams[0]].kpts_to_numpy()):.2f}%)"
        )

    feats[cams[0]].filter_feature_by_mask(inlMask, verbose=True)
    feats[cams[1]].filter_feature_by_mask(inlMask, verbose=True)

    if viz_results:
        fig, axes = plt.subplots(1, 2)
        for ax, cam in zip(axes, cams):
            icepy_viz.plot_features(
                images[cam].read_image(epoch).value, feats[cam], ax=ax
            )
        fig.set_size_inches((18.5, 10.5))
        fig.savefig("test_out/test_matched_keypoints.png", dpi=300)
        plt.close(fig)

        icepy_viz.plot_matches(
            images[cams[0]].read_image(epoch).value,
            images[cams[1]].read_image(epoch).value,
            feats[cams[0]].kpts_to_numpy(),
            feats[cams[1]].kpts_to_numpy(),
            path="test_out/test_fullres.png",
        )

    return feats
