from pathlib import Path
import numpy as np
import matplotlib.cm as cm
import torch
import json

from ..thirdparty.SuperGluePretrainedNetwork.models.matching import Matching
from ..thirdparty.SuperGluePretrainedNetwork.models.utils import (
    make_matching_plot,
    AverageTimer,
    read_image,
    frame2tensor,
    vizTileRes,
)

torch.set_grad_enabled(False)


def match_pair(pair, image0, image1, maskBB, opt):

    # @TODO: implement all checks...

    # Parameters
    do_viz = opt["viz"]

    # Load the SuperPoint and SuperGlue models.
    device = "cuda" if torch.cuda.is_available() and not opt["force_cpu"] else "cpu"
    print('Running inference on device "{}"'.format(device))
    config = {
        "superpoint": {
            "nms_radius": opt["nms_radius"],
            "keypoint_threshold": opt["keypoint_threshold"],
            "max_keypoints": opt["max_keypoints"],
        },
        "superglue": {
            "weights": opt["superglue"],
            "sinkhorn_iterations": opt["sinkhorn_iterations"],
            "match_threshold": opt["match_threshold"],
        },
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    output_dir = Path(opt["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory "{}"'.format(output_dir))
    if opt["viz"]:
        print("Will write visualization images to", 'directory "{}"'.format(output_dir))

    timer = AverageTimer()
    name0, name1 = pair[:2]
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    matches_path = output_dir / "{}_{}_matches.npz".format(stem0, stem1)
    viz_path = output_dir / "{}_{}_matches.{}".format(
        stem0, stem1, opt["viz_extension"]
    )

    # Convert images to tensors
    inp0 = frame2tensor(image0, device)
    inp1 = frame2tensor(image1, device)

    # Perfomr the matching
    pred = matching({"image0": inp0, "image1": inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

    # Retrieve results
    kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
    descriptors0, descriptors1 = pred["descriptors0"], pred["descriptors1"]
    scores0, scores1 = pred["scores0"], pred["scores1"]
    matches0 = pred["matches0"]
    conf = pred["matching_scores0"]
    valid = matches0 > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches0[valid]]
    descriptors0 = descriptors0[:, valid]
    descriptors1 = descriptors1[:, matches0[valid]]
    scores0 = scores0[valid]
    scores1 = scores1[matches0[valid]]
    conf = conf[valid]

    # Visualize the matches.
    if do_viz:
        color = cm.jet(mconf)
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
            "Matches: {}".format(len(mkpts0)),
        ]

        # Display extra parameter info.
        k_thresh = matching.superpoint.config["keypoint_threshold"]
        m_thresh = matching.superglue.config["match_threshold"]
        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
            "Image Pair: {}:{}".format(stem0, stem1),
        ]

        make_matching_plot(
            image0,
            image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            viz_path,
            opt["show_keypoints"],
            opt["fast_viz"],
            opt["opencv_display"],
            "Matches",
            small_text,
        )

        timer.update("viz_match")


if __name__ == "__main__":
    print("test matching")

    from tiles import Tiles
    from classes.images import Image, ImageDS

    matching_config = "config/opt_matching.json"
    with open(
        matching_config,
    ) as f:
        opt_matching = json.load(f)

    images = ImageDS(Path("images"))

    img0, img1 = images[0], images[1]
    im0_path, im1_path = images.get_image_path(0), images.get_image_path(1)

    pair = [im0_path, im1_path]

    maskBB = []

    match_pair(pair, img0, img1, maskBB, opt_matching)
