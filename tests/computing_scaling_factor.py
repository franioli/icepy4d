'''OLD'''
# Camera baseline
baseline_world = np.linalg.norm(
    cfg.georef.camera_centers_world[0] - cfg.georef.camera_centers_world[1]
)

# cam_baseline = np.linalg.norm(
#     cameras[cams[0]][0].get_C_from_pose() -
#     cameras[cams[1]][0].get_C_from_pose()
# )
# scale_fct = baseline_world / cam_baseline
# scale_fct = 261.60624502293524

# Fix the EO of both the cameras as those estimated in the first epoch
# if epoch > 0:
#     for cam in cams:
#         cameras[cam][epoch] = cameras[cam][0]
#     print('Camera exterior orientation fixed to that of the master cameras.')

""" Compute scale factor .... to be placed in two_view_geometry class"""
# Perform Relative orientation of the two cameras in arbitrary local RS
cameras = dict.fromkeys(cams)
cameras[cams[0]], cameras[cams[1]] = [], []
for cam in cams:
    cameras[cam].append(
        Camera(
            width=im_width,
            height=im_height,
            calib_path=cfg.paths.caldir / f'{cam}.txt'
        )
    )
relative_ori = Two_view_geometry(
    [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
    [features[cams[0]][epoch].get_keypoints(),
        features[cams[1]][epoch].get_keypoints()],
)
relative_ori.relative_orientation(
    threshold=1.5,
    confidence=0.999999,
)
cameras[cams[1]][epoch] = relative_ori.cameras[1]

'''Compute scale factor from targets'''
# Read Targets 
target_paths = ["data/target_image_p1_all.csv", "data/target_image_p2_all.csv"]
targets = [Targets(
    im_file_path=target_paths,
    obj_file_path="data/target_world_all.csv"), ]

# Extract targets for compute world-local scale factor
targets_to_extract = ["F2", "F3"]
im_coord = {}
for id, cam in enumerate(cams):
    im_coord[cam] = targets[0].extract_image_coor_by_label(
        targets_to_extract, id)
targets_world = targets[0].extract_object_coor_by_label(targets_to_extract)

# Triangulate targets
triangulation = Triangulate# Read Targets
target_paths = ["data/target_image_p1_all.csv", "data/target_image_p2_all.csv"]
targets = [Targets(
    im_file_path=target_paths,
    obj_file_path="data/target_world_all.csv"), ]

# Extract targets for compute world-local scale factor
targets_to_extract = ["F2", "T2"]
im_coord = {}
for id, cam in enumerate(cams):
    im_coord[cam] = targets[0].extract_image_coor_by_label(
        targets_to_extract, id)
targets_world = targets[0].extract_object_coor_by_label(targets_to_extract)

# Triangulate targets
triangulation = Triangulate(
    [cameras[cams[0]][epoch],
     cameras[cams[1]][epoch]],
    [im_coord[cams[0]],
     im_coord[cams[1]]]
)
targets_local = triangulation.triangulate_two_views()

# Compute baselines and scale factor
baseline_local = np.linalg.norm(
    targets_local[0] - targets_local[1]
)
baseline_world = np.linalg.norm(
    targets_world[0] - targets_world[1]
)
(
    [cameras[cams[0]][epoch],
     cameras[cams[1]][epoch]],
    [im_coord[cams[0]],
     im_coord[cams[1]]]
)
targets_local = triangulation.triangulate_two_views()

# Compute baselines and scale factor
baseline_local = np.linalg.norm(
    targets_local[0] - targets_local[1]
)
baseline_world = np.linalg.norm(
    targets_world[0] - targets_world[1]
)


# """Compute scale factor from perspective centers from CALGE"""
# C0 = np.array([151.8827,99.1017,91.49643])
# C1 = np.array([328.4277,302.5639,135.11935])
# baseline_world = np.linalg.norm(
#     C0 - C1
# )

# baseline_local = np.linalg.norm(
#     cameras[cams[0]][0].get_C_from_pose() -
#     cameras[cams[1]][0].get_C_from_pose()
# )


scale_fct = baseline_world / baseline_local
print(f"Baseline in world ref system: {baseline_world}")
print(f"Baseline in local ref system: {baseline_local}")
print(f"Computed world-local scale factor: {scale_fct}")