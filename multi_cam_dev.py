# Import required standard modules
from pathlib import Path

import numpy as np

# Import required icepy4d4D modules
from icepy4d import core as icecore
from icepy4d.core import Epoch
from icepy4d import matching
from icepy4d import sfm
from icepy4d import io
from icepy4d import utils
from icepy4d.metashape import metashape as MS
from icepy4d.utils import initialization


# Parse the configuration file
CFG_FILE = "config/config_2022.yaml"

# Parse the configuration file
cfg_file = Path(CFG_FILE)
cfg = initialization.parse_cfg(cfg_file, ignore_errors=True)

# Initialize the logger
logger = utils.get_logger()

# Get the list of cameras from the configuration file
cams = ["cam0", "cam1", "cam2"]

root_dir = Path("./sandbox/test_multi_cam")

# Build a dictionary of images containing the name of the cameras as keys and Image objects as values
im_epoch = {
    cams[0]: icecore.Image(root_dir / "DJI_20220726095348_0003.JPG"),
    cams[1]: icecore.Image(root_dir / "DJI_20220726095358_0004.JPG"),
    cams[2]: icecore.Image(root_dir / "DJI_20220726095421_0005.JPG"),
}

# Get epoch timestamp as the timestamp of the first image and define epoch directory
epoch_timestamp = im_epoch[cams[0]].timestamp

# Load cameras
cams_ep = {}
for cam in cams:
    calib = icecore.Calibration(root_dir / "dji_p1_jpg_belvedere2022.xml")
    cams_ep[cam] = calib.to_camera()


# Create empty features
feat_ep = {cam: icecore.Features() for cam in cams}

# Create the epoch object
epoch = Epoch(
    timestamp=epoch_timestamp,
    images=im_epoch,
    cameras=cams_ep,
    features=feat_ep,
    epoch_dir='./',
)
print(f"Epoch: {epoch}")

# Match first two imgs
matcher = matching.LightGlueMatcher()
matcher.match(
    epoch.images[cams[0]].value,
    epoch.images[cams[1]].value,
    quality=matching.Quality.HIGH,
    tile_selection= matching.TileSelection.PRESELECTION,
    grid=[2, 3],
    overlap=200,
    origin=[0, 0],
    do_viz_matches=True,
    do_viz_tiles=True,
    min_matches_per_tile = 3,
    max_keypoints = 8196,    
    save_dir='./matches',
    geometric_verification=matching.GeometricVerification.PYDEGENSAC,
    threshold=2,
    confidence=0.9999,
)

# Define a dictionary with empty Features objects for each camera, which will be filled with the matched keypoints, descriptors and scores
f = {cam: icecore.Features() for cam in cams}

# Stack matched keypoints, descriptors and scores into Features objects
f[cams[0]].append_features_from_numpy(
    x=matcher.mkpts0[:, 0],
    y=matcher.mkpts0[:, 1],
    descr=matcher.descriptors0,
    scores=matcher.scores0,
)
f[cams[1]].append_features_from_numpy(
    x=matcher.mkpts1[:, 0],
    y=matcher.mkpts1[:, 1],
    descr=matcher.descriptors1,
    scores=matcher.scores1,
)

# Store the dictionary with the features in the Epoch object
epoch.features = f


# Incremental reconstruction

# Compute the camera baseline from a-priori camera positions
baseline = np.linalg.norm(
    cfg.georef.camera_centers_world[0] - cfg.georef.camera_centers_world[1]
)

# Initialize RelativeOrientation class with a list containing the two
# cameras and a list contaning the matched features location on each camera.
relative_ori = sfm.RelativeOrientation(
    [epoch.cameras[cams[0]], epoch.cameras[cams[1]]],
    [
        epoch.features[cams[0]].kpts_to_numpy(),
        epoch.features[cams[1]].kpts_to_numpy(),
    ],
)

# Estimate the relative orientation
relative_ori.estimate_pose(
    threshold=cfg.matching.pydegensac_threshold,
    confidence=0.999999,
    scale_factor=baseline,
)

# Store result in camera 1 object
epoch.cameras[cams[1]] = relative_ori.cameras[1]


triang = sfm.Triangulate(
    [epoch.cameras[cams[0]], epoch.cameras[cams[1]]],
    [
        epoch.features[cams[0]].kpts_to_numpy(),
        epoch.features[cams[1]].kpts_to_numpy(),
    ],
)
points3d = triang.triangulate_two_views(
    compute_colors=True, image=epoch.images[cams[1]].value, cam_id=1
)

# Update timer

# Extract the image coordinates of the targets from the Targets object
image_coords = [
    epoch.targets.get_image_coor_by_label(cfg.georef.targets_to_use, cam_id=id)[0] for id, cam in enumerate(cams)
]
print(f"Targets coordinates on image 0:\n{image_coords[0]}")
print(f"Targets coordinates on image 1:\n{image_coords[1]}")

obj_coords = epoch.targets.get_object_coor_by_label(cfg.georef.targets_to_use)[0]
print(f"Targets coordinates in object space:\n{obj_coords}")

# Initialize AbsoluteOrientation object with a list containing the two
abs_ori = sfm.Absolute_orientation(
    (epoch.cameras[cams[0]], epoch.cameras[cams[1]]),
    points3d_final=obj_coords,
    image_points=image_coords,
    camera_centers_world=cfg.georef.camera_centers_world,
)

# Estimate the absolute orientation transformation
T = abs_ori.estimate_transformation_linear(estimate_scale=True)

# Transform the 3D points
points3d = abs_ori.apply_transformation(points3d=points3d)

# Store the absolute orientation transformation in the camera objects
for i, cam in enumerate(cams):
    epoch.cameras[cam] = abs_ori.cameras[i]

# Convert the 3D points to an icepy4d Points object
pts = icecore.Points()
pts.append_points_from_numpy(
    points3d,
    track_ids=epoch.features[cams[0]].get_track_ids(),
    colors=triang.colors,
)

# Store the points in the Epoch object
epoch.points = pts

# Update timer

# Save epoch as a pickle object
if epoch.save_pickle(f"{epoch.epoch_dir}/{epoch}.pickle"):
    logger.info(f"{epoch} saved successfully")
else:
    logger.error(f"Unable to save {epoch}")