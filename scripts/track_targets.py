import shutil
from pathlib import Path

from icepy4d.core import Image, Targets
from icepy4d.utils import get_logger
from icepy4d.utils.track_targets import TrackTargets

logger = get_logger(__name__)

# Define input parameters
cams = ["p1", "p2"]
IM_DIRS = ["data/img/p1", "data/img/p2"]
MASTER_IMAGES = [
    "p1_20230725_135953_IMG_1149.JPG",
    "p2_20230725_140026_IMG_0887.JPG",
]
TARGETS_DIR = "data/targets"
TARGETS_WORLD_FNAME = "targets_world.csv"
TARGETS_IMAGES_FNAMES = [
    "p1_20230725_135953_IMG_1149.csv",
    "p2_20230725_140026_IMG_0887.csv",
]
OUT_DIR = "res/target_tracking"

# Define targets to track
targets_to_track = ["F2", "F10", "T3"]

# Template matching parameters
template_width = 32
search_width = 128

for id, cam in enumerate(cams):
    # Build Targets object
    target_dir = Path(TARGETS_DIR)
    targets = Targets(
        im_file_path=[target_dir / f for f in TARGETS_IMAGES_FNAMES],
        obj_file_path=target_dir / TARGETS_WORLD_FNAME,
    )

    # Build list of Image objects
    img_dir = Path(IM_DIRS[id])
    images = [Image(f) for f in sorted(img_dir.glob("*"))]
    master = img_dir / MASTER_IMAGES[id]
    out_dir = Path(OUT_DIR) / cam
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Define nx2 array with image coordinates of the targets to track
    # in the form of:
    # [x1, y1],
    # [x2, y2]...
    # You can create it manually or use Target class
    targets_coord, _ = targets.get_image_coor_by_label(targets_to_track, id)

    # Define TrackTarget object and run tracking
    logger.info(f"Tracking targets {targets_to_track} in camera {cam}")
    tracking = TrackTargets(
        master=master,
        images=images,
        targets=targets_coord.squeeze(),
        out_dir=out_dir,
        target_names=targets_to_track,
        template_width=template_width,
        search_width=search_width,
        verbose=True,
        viz_tracked=True,
        parallel=False,
    )
    tracking.track()

print("Done.")
