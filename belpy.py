import numpy as np
import cv2
import pickle
from pathlib import Path
from easydict import EasyDict as edict

from lib.classes import Camera, Imageds, Features, Targets
from lib.matching.matching_base import MatchingAndTracking
from lib.sfm.two_view_geometry import Two_view_geometry
from lib.sfm.triangulation import Triangulate
from lib.sfm.absolute_orientation import (
    Absolute_orientation,
    Space_resection,
)
from lib.read_config import parse_yaml_cfg
from lib.point_clouds import (
    create_point_cloud,
    write_ply,
)
from lib.utils import (
    create_directory,
    AverageTimer,
    build_dsm,
    generate_ortophoto,
)
from lib.visualization import display_point_cloud
from lib.import_export.export2bundler import write_bundler_out

from thirdparty.transformations import euler_from_matrix, euler_matrix

from lib.metashape.metashape import MetashapeProject

timer_global = AverageTimer(newline=True)
root_path = Path().absolute()

# Parameters to be put in option yaml
last_match_path = root_path / "res/last_epoch/last_features.pickle"
do_export_to_bundler = True
do_metashape_bba = True
do_metashape_dense = True
targets_to_use = ["F2", "F4"]  # 'T4',
pydegensac_treshold = 1

# Parse options from yaml file
cfg_file = root_path / "config/config_base.yaml"
cfg = parse_yaml_cfg(cfg_file)

""" Inizialize Variables """
# @TODO: put this in an inizialization function
cams = cfg.paths.camera_names
features = dict.fromkeys(cams)
cams = cfg.paths.camera_names
for cam in cams:
    features[cam] = []

# Create Image Datastore objects
images = dict.fromkeys(cams)
for cam in cams:
    images[cam] = Imageds(cfg.paths.image_dir / cam)

# Read target image coordinates and object coordinates
targets = []
for epoch in cfg.proc.epoch_to_process:

    p1_path = cfg.georef.target_dir / (
        images[cams[0]].get_image_stem(epoch) + cfg.georef.target_file_ext
    )

    p2_path = cfg.georef.target_dir / (
        images[cams[1]].get_image_stem(epoch) + cfg.georef.target_file_ext
    )

    targets.append(
        Targets(
            im_file_path=[p1_path, p2_path], obj_file_path="data/target_world_p1.csv"
        )
    )

# Cameras
# @TODO: build function for variable inizialization
cameras = dict.fromkeys(cams)
cameras[cams[0]], cameras[cams[1]] = [], []
im_height, im_width = 4000, 6000
# @TODO: store this information in exif inside an Image Class
point_clouds = []

""" Big Loop over epoches """

print("Processing started:")
print("-----------------------")
timer = AverageTimer(newline=True)
for epoch in cfg.proc.epoch_to_process:

    print(f"\nProcessing epoch {epoch}...")

    epochdir = Path(cfg.paths.results_dir) / f"epoch_{epoch}"

    """Perform matching and tracking"""
    if cfg.proc.do_matching:
        features = MatchingAndTracking(
            cfg=cfg,
            epoch=epoch,
            images=images,
            features=features,
        )
    elif not features[cams[0]]:
        try:
            with open(last_match_path, "rb") as f:
                features = pickle.load(f)
                print("Loaded previous matches")
        except:
            print(
                f"Features not found in {str(last_match_path)}. Please enable performing matching or provide valid path to already computed matches."
            )
    else:
        print("Features already loaded.")
    timer.update("matching")

    """ SfM """

    print(f"Reconstructing epoch {epoch}...")

    # Initialize Intrinsics
    # Inizialize Camera Intrinsics at every epoch setting them equal to the those of the reference cameras.
    # @TODO: replace append with insert or a more robust data structure...
    for cam in cams:
        cameras[cam].append(
            Camera(
                width=im_width,
                height=im_height,
                calib_path=cfg.paths.calibration_dir / f"{cam}.txt",
            )
        )

    # --- Space resection of Master camera ---#
    # At the first epoch, perform Space resection of the first camera by using GCPs. At all other epoches, set camera 1 EO equal to first one.
    # if epoch == 0:
    #     ''' Initialize Single_camera_geometry class with a cameras object'''
    #     targets_to_use = ['T2','T3','T4','F2' ]
    #     space_resection = Space_resection(cameras[cams[0]][epoch])
    #     space_resection.estimate(
    #         targets[epoch].extract_image_coor_by_label(targets_to_use,cam_id=0),
    #         targets[epoch].extract_object_coor_by_label(targets_to_use)
    #         )
    #     # Store result in camera 0 object
    #     cameras[cams[0]][epoch] = space_resection.camera
    # else:
    #     cameras[cams[0]][epoch] = cameras[cams[0]][0]

    # --- Perform Relative orientation of the two cameras ---#
    # Initialize Two_view_geometry class with a list containing the two cameras and a list contaning the matched features location on each camera.
    relative_ori = Two_view_geometry(
        [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
        [
            features[cams[0]][epoch].get_keypoints(),
            features[cams[1]][epoch].get_keypoints(),
        ],
    )
    relative_ori.relative_orientation(
        threshold=pydegensac_treshold,
        confidence=0.999999,
        scale_factor=np.linalg.norm(
            cfg.georef.camera_centers_world[0] - cfg.georef.camera_centers_world[1]
        ),
    )
    # Store result in camera 1 object
    cameras[cams[1]][epoch] = relative_ori.cameras[1]

    # --- Triangulate Points ---#
    # Initialize a Triangulate class instance with a list containing the two cameras and a list contaning the matched features location on each camera. Triangulated points are saved as points3d proprierty of the Triangulate object (eg., triangulation.points3d)
    triangulation = Triangulate(
        [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
        [
            features[cams[0]][epoch].get_keypoints(),
            features[cams[1]][epoch].get_keypoints(),
        ],
    )
    points3d = triangulation.triangulate_two_views()
    triangulation.interpolate_colors_from_image(
        images[cams[1]][epoch],
        cameras[cams[1]][epoch],
        convert_BRG2RGB=True,
    )

    # --- Absolute orientation (-> coregistration on stable points) ---#
    if cfg.proc.do_coregistration:
        abs_ori = Absolute_orientation(
            (cameras[cams[0]][epoch], cameras[cams[1]][epoch]),
            points3d_final=targets[epoch].extract_object_coor_by_label(targets_to_use),
            image_points=(
                targets[epoch].extract_image_coor_by_label(targets_to_use, cam_id=0),
                targets[epoch].extract_image_coor_by_label(targets_to_use, cam_id=1),
            ),
            camera_centers_world=cfg.georef.camera_centers_world,
        )
        T = abs_ori.estimate_transformation_linear(estimate_scale=True)
        points3d = abs_ori.apply_transformation(points3d=points3d)
        for i, cam in enumerate(cams):
            cameras[cam][epoch] = abs_ori.cameras[i]

    # Create point cloud and save .ply to disk
    pcd_epc = create_point_cloud(points3d, triangulation.colors)

    # Filter outliers in point cloud with SOR filter
    if cfg.other.do_SOR_filter:
        _, ind = pcd_epc.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
        pcd_epc = pcd_epc.select_by_index(ind)
        print("Point cloud filtered by Statistical Oulier Removal")

    # Write point cloud to disk and store it in Point Cloud List
    # write_ply(pcd_epc, f"res/pt_clouds/sparse_pts_t{epoch}.ply")
    write_ply(pcd_epc, epochdir / f"sparse_pts_t{epoch}.ply")
    point_clouds.append(pcd_epc)

    # Export results in Bundler format
    if do_export_to_bundler:
        write_bundler_out(
            export_dir=epochdir / "metashape",
            epoches=[epoch],
            images=images,
            cams=cams,
            cameras=cameras,
            features=features,
            point_clouds=point_clouds,
            targets=targets,
            targets_to_use=targets_to_use,
        )

    timer.update("relative orientation")

    # Metashape BBA and dense cloud
    """" TO be organized!"""
    if do_metashape_bba:
        ms_dir = Path(root_path / f"res/epoch_{epoch}/metashape")
        ms_cfg = edict(
            {
                "project_name": ms_dir / f"belpy_epoch_{epoch}.psx",
                "im_path": ms_dir / "data/images/",
                "bundler_file_path": ms_dir / f"data/belpy_epoch_{epoch}.out",
                "bundler_im_list": ms_dir / "data/im_list.txt",
                "gcp_filename": ms_dir / "data/gcps.txt",
                "calib_filename": [
                    root_path / "res/calib_metashape/belpy_35mm_280722_selfcal_all.xml",
                    root_path / "res/calib_metashape/belpy_24mm_280722_selfcal_all.xml",
                ],
                "im_ext": "jpg",
                "camera_location": [
                    [309.261, 301.051, 135.008],  # IMG_1289
                    [151.962, 99.065, 91.643],  # IMG_2814
                ],
                "gcp_accuracy": [0.01, 0.01, 0.01],
                "cam_accuracy": [0.001, 0.001, 0.001],
                "prm_to_fix": [
                    "Cx",
                    "Cy",
                    "B1",
                    "B2",
                    "K1",
                    "K2",
                    "K3",
                    "K4",
                    "P1",
                    "P2",
                ],
                "optimize_cameras": True,
                "build_dense": True,
                "dense_path": ms_dir,
                "dense_name": f"dense_epoch_{epoch}.ply",
                "force_overwrite_projects": True,
            }
        )

        ms = MetashapeProject(ms_cfg, timer)
        ms.process_full_workflow()
        # timer.update("bundle and dense")

    timer.print(f"Epoch {epoch} completed")
    timer_global.update(f"epoch {epoch}")


timer_global.print("All epoches completed")

# Visualize point cloud
display_point_cloud(
    point_clouds,
    [cameras[cams[0]][epoch], cameras[cams[1]][epoch]],
    plot_scale=10,
)


""" Compute DSM and orthophotos """
# @TODO: implement better DSM class

compute_orthophoto_dsm = False
if compute_orthophoto_dsm:
    print("DSM and orthophoto generation started")
    res = 0.03
    xlim = [-100.0, 80.0]
    ylim = [-10.0, 65.0]

    dsms = []
    ortofoto = dict.fromkeys(cams)
    ortofoto[cams[0]], ortofoto[cams[1]] = [], []
    for epoch in cfg.proc.epoch_to_process:
        print(f"Epoch {epoch}")
        dsms.append(
            build_dsm(
                np.asarray(point_clouds[epoch].points),
                dsm_step=res,
                xlim=xlim,
                ylim=ylim,
                make_dsm_plot=False,
                # fill_value = ,
                # save_path=f'res/dsm/dsm_app_epoch_{epoch}.tif'
            )
        )
        print("DSM built.")
        for cam in cams:
            fout_name = f"res/ortofoto/ortofoto_app_cam_{cam}_epc_{epoch}.tif"
            ortofoto[cam].append(
                generate_ortophoto(
                    cv2.cvtColor(images[cam][epoch], cv2.COLOR_BGR2RGB),
                    dsms[epoch],
                    cameras[cam][epoch],
                    xlim=xlim,
                    ylim=ylim,
                    save_path=fout_name,
                )
            )
        print("Orthophotos built.")
