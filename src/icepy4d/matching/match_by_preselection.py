import logging
import numpy as np
import cv2
import pickle

from matplotlib import pyplot as plt
from typing import Tuple, List, Union
from pathlib import Path
from easydict import EasyDict as edict

import icepy4d.classes as icepy4d_classes
import icepy4d.visualization as icepy_viz
from icepy4d.classes.images import read_image
from icepy4d.matching.superglue_matcher import SuperGlueMatcher
from icepy4d.matching.utils import geometric_verification

from scipy.spatial import KDTree
from scipy.cluster.vq import kmeans, vq, whiten
from scipy.spatial.distance import pdist
from icepy4d.utils.timer import AverageTimer

DEFAULT_PATCH_SIZE = 1000


def average_distance(points: np.ndarray):
    """Calculate the pairwise distances between all points and Return the mean of the distances"""
    return pdist(points).mean()


def find_centroids_kmeans(
    data: np.ndarray, n_cluster: int, viz_clusters: bool = False
) -> Tuple[np.ndarray]:
    """
    Applies the k-means algorithm to cluster the input data into n_cluster groups and returns the centroids and the
    classes to which each point belongs.

    Args:
        data: The input data as a 2D NumPy array of shape (n_samples, n_features).
        n_cluster: The desired number of clusters.
        viz_clusters: Whether to plot the data points and the cluster centroids. Default is False.

    Returns:
        A tuple containing:
            - centroids: A 2D NumPy array of shape (n_cluster, n_features) representing the centroids of the clusters.
            - classes: A 1D NumPy array of length n_samples containing the class assignment for each data point.
    """
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


def find_matches_on_patches(
    images: icepy4d_classes.ImagesDict,
    patches_lim: dict,
    epoch: int,
    features: icepy4d_classes.FeaturesDict,
    cfg: dict,
    do_geometric_verification: bool = True,
    geometric_verification_threshold: float = 5,
    viz_results: bool = True,
    fast_viz: bool = True,
    viz_path: Union[Path, str] = None,
):
    """
    Find and verify matches between patches of two images.

    Args:
        images (icepy4d_classes.ImagesDict): A dictionary of `ImagesDict` instances, where each instance represents an image captured by a specific camera.
        patches_lim (dict): A dictionary containing the patch limits (x1, y1, x2, y2) for each camera, where x1 and y1 are the coordinates of the top-left corner of the patch, and x2 and y2 are the coordinates of the bottom-right corner of the patch.
        epoch (int): An integer indicating the epoch number.
        features (icepy4d_classes.FeaturesDict): A dictionary of `FeaturesDict` instances, where each instance represents the detected features in an image captured by a specific camera.
        cfg (dict): A dictionary containing the configuration parameters for the matching algorithm.
        viz_results (bool, optional): A boolean indicating whether to visualize the matched features. Defaults to True.
        fast_viz (bool, optional): A boolean indicating whether to use fast visualization. Defaults to True.
        viz_path (Union[Path, str], optional): A path to the directory where the visualization results will be saved. Defaults to None.
    """

    logging.info("Geometric verification of matches on full images patches...")
    cams = list(features.keys())
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

    matcher = SuperGlueMatcher(cfg)
    mkpts = matcher.match(patches[cams[0]], patches[cams[1]])
    if do_geometric_verification:
        mkpts = matcher.geometric_verification(
            threshold=geometric_verification_threshold,
            confidence=0.99,
            symmetric_error_check=False,
        )
    if viz_results:
        matcher.viz_matches(
            viz_path,
            fast_viz=fast_viz,
        )

    mkpts = {
        cams[0]: matcher.mkpts0 + np.array(patches_lim[cams[0]][0:2]).astype(np.int32),
        cams[1]: matcher.mkpts1 + np.array(patches_lim[cams[1]][0:2]).astype(np.int32),
    }
    descriptors = {cams[0]: matcher.descriptors0, cams[1]: matcher.descriptors1}
    scores = {cams[0]: matcher.scores0, cams[1]: matcher.scores1}
    for cam in cams:
        features[cam].append_features_from_numpy(
            x=mkpts[cam][:, 0],
            y=mkpts[cam][:, 1],
            descr=descriptors[cam],
            scores=scores[cam],
        )


def match_by_preselection(
    images: icepy4d_classes.ImagesDict,
    features: icepy4d_classes.FeaturesDictEpoch,
    camera_names: List[str],
    epoch: int,
    cfg: edict,
    out_dir: Union[Path, str],
    n_tiles: int = 8,
    n_dist: int = 2,
    viz_results: bool = True,
    fast_viz: bool = True,
) -> icepy4d_classes.FeaturesDictEpoch:
    """
    Matches features between two images by preselecting keypoints on downscaled images, and then refining the matches on full-resolution patches of the images.

    Args:
        images (icepy4d_classes.ImagesDict): A dictionary of image paths for each camera.
        features (icepy4d_classes.FeaturesDictEpoch): A dictionary of features for each camera.
        epoch (int): The index of the epoch to process.
        cfg (edict): A dictionary with configuration parameters.
        n_tiles (int, optional): The number of tiles to divide the images into. Defaults to 8.
        n_dist (int, optional): The number of times the average point distance of each cluster has to be multiplied with to get the tile size. Defaults to 2.
        viz_results (bool, optional): Whether to visualize the results. Defaults to True.
        fast_viz (bool, optional): Make plot faster with opencv (it uses matplotlib otherwise). Defaults to True.
    Returns:
        icepy4d_classes.FeaturesDictEpoch: A dictionary of refined features for each camera.
    """

    if not isinstance(cfg, dict):
        raise TypeError("opt must be a dictionary")
    required_keys = [
        "weights",
        "keypoint_threshold",
        "max_keypoints",
        "match_threshold",
        "force_cpu",
    ]
    missing_keys = [key for key in required_keys if key not in cfg]
    if missing_keys:
        raise KeyError(
            f"Missing required keys: {', '.join(missing_keys)} in SuperGlue Matcher option dictionary"
        )

    cams = camera_names
    im_paths = [images[cam].get_image_path(epoch) for cam in cams]
    im_stems = [images[cam].get_image_stem(epoch) for cam in cams]
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    timer = AverageTimer()

    logging.info("Running matching on downscaled images")
    image0, scales = read_image(im_paths[0], resize=[2004, 1336])
    image1, scales = read_image(im_paths[1], resize=[2004, 1336])

    sg_opt = {
        "weights": cfg.weights,
        "keypoint_threshold": 0.0001,
        "max_keypoints": cfg.max_keypoints,
        "match_threshold": 0.05,
        "force_cpu": False,
    }
    matcher = SuperGlueMatcher(sg_opt)
    match_presel = matcher.match(image0, image1)
    match_presel = matcher.geometric_verification(
        threshold=10, confidence=0.9, symmetric_error_check=False
    )
    if viz_results:
        matcher.viz_matches(
            path=out_dir / f"{im_stems[0]}_{im_stems[1]}_matches_low_res.png",
            fast_viz=fast_viz,
        )

    match_presel = {cams[i]: match_presel[i] * scales[i] for i, _ in enumerate(cams)}
    centroids, classes = find_centroids_kmeans(match_presel[cams[0]], n_tiles)

    # Add fake controid to compute
    new_nodes = np.array([[4300.0, 2000.0], [4300.0, 2800.0]])
    centroids = np.append(centroids, new_nodes, axis=0)

    fig, axes = plt.subplots(1, 2)
    for ax, cam in zip(axes, cams):
        icepy_viz.plot_points(
            images[cam].read_image(epoch).value,
            match_presel[cam],
            ax=ax,
        )
    fig.set_size_inches((18.5, 10.5))
    fig.savefig(out_dir / f"{im_stems[0]}_{im_stems[1]}_kpts_low_res.png", dpi=300)
    plt.close(fig)
    timer.update("preselection")

    logging.info("Running matching on full resolutions patches")
    features = {cam: icepy4d_classes.Features() for cam in cams}
    for i, center in enumerate(centroids):
        logging.info(f"Processing tile {i}")
        try:
            win_size = int(
                n_dist * average_distance(match_presel[cams[0]][classes == i, :])
            )
        except:
            logging.warning(
                f"Unable to compute patch size automatically. Using deafult patch size of {2*DEFAULT_PATCH_SIZE}"
            )
            win_size = DEFAULT_PATCH_SIZE

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
        find_matches_on_patches(
            images,
            patches_lim,
            epoch,
            features,
            cfg,
            viz_results,
            fast_viz,
            viz_path=out_dir / f"{im_stems[0]}_{im_stems[1]}_matches_patch_{i}.png",
        )
    logging.info("Matching by patches completed.")
    timer.update("matching by patches")

    geometric_verification(
        features,
        threshold=cfg.pydegensac_threshold,
        confidence=cfg.pydegensac_confidence,
    )
    timer.update("Geometric verification")

    if viz_results:
        fig, axes = plt.subplots(1, 2)
        for ax, cam in zip(axes, cams):
            icepy_viz.plot_features(
                images[cam].read_image(epoch).value,
                features[cam],
                ax=ax,
            )
        fig.set_size_inches((18.5, 10.5))
        fig.savefig(out_dir / f"{im_stems[0]}_{im_stems[1]}_kpts_full_res.png", dpi=300)
        plt.close(fig)

        icepy_viz.plot_matches(
            images[cams[0]].read_image(epoch).value,
            images[cams[1]].read_image(epoch).value,
            features[cams[0]].kpts_to_numpy(),
            features[cams[1]].kpts_to_numpy(),
            path=str(out_dir / f"{im_stems[0]}_{im_stems[1]}_matches_full_res.png"),
        )

    # Write matched points to disk
    for jj, cam in enumerate(cams):
        features[cam].save_as_txt(out_dir / f"{im_stems[jj]}_mktps.txt")

    # Save current epoch features as pickle file
    fname = out_dir / f"{im_stems[0]}_{im_stems[1]}_features.pickle"
    with open(fname, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Matching completed")
    timer.update("export")
    timer.print("matching")

    return features
