import numpy as np
import open3d as o3d

from typing import List
from pathlib import Path
from easydict import EasyDict as edict
from tqdm import tqdm
from random import randint

from ..classes.images import Image, ImageDS
from ..utils.initialization import parse_yaml_cfg


def read_asci_pc(path):
    data = np.loadtxt(path, delimiter=",")
    pts = data[:, 0:3]
    # cols = data[:, 3:6]
    # col = np.random.randint(0, 255, (1, 3)) / 255
    col = np.array([255, 0, 0]).reshape(1, 3) / 255
    cols = np.repeat(col, pts.shape[0], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    return pcd


def viz_point_cloud(
    pcd: str,
    win_size: List[int] = None,
):
    vis = o3d.visualization.Visualizer()
    if win_size:
        vis.create_window(width=win_size[0], height=win_size[1])
    else:
        vis.create_window()
    # pcd = o3d.io.read_point_cloud(path, format=format)
    vis.add_geometry(pcd)
    # vis.destroy_window()


def set_view_options(
    vis,
    view_dict: dict = None,
    render_opt_json: str = None,
) -> None:

    ctr = vis.get_view_control()
    ctr.set_front(view_dict.front)
    ctr.set_lookat(view_dict.lookat)
    ctr.set_up(view_dict.up)
    ctr.set_zoom(view_dict.zoom)
    render_opt = vis.get_render_option()
    render_opt.load_from_json(render_opt_json)


def viz_loop(epoch_dict, view, o3d_render_opt_fname, out_dir):

    vis = o3d.visualization.Visualizer()
    vis.create_window()  # width=640, height=480)
    pcd = o3d.io.read_point_cloud("res/point_clouds/dense_ep_0_2022_07_28.ply")
    set_view_options(vis, view, o3d_render_opt_fname)

    vis.add_geometry(pcd)
    # vis.run()
    vis.poll_events()
    vis.update_renderer()

    for epoch in tqdm(cfg.proc.epoch_to_process):
        # print(f"Epoch {epoch}")
        fname = f"res/point_clouds/dense_ep_{epoch}_{epoch_dict[epoch]}.ply"
        pcd.points = o3d.io.read_point_cloud(fname).points
        pcd.colors = o3d.io.read_point_cloud(fname).colors

        vis.update_geometry(pcd)
        set_view_options(vis, view, o3d_render_opt_fname)
        vis.poll_events()
        vis.update_renderer()
        screen_name = str(out_dir / f"ep_{epoch}_{epoch_dict[epoch]}.png")
        vis.capture_screen_image(screen_name)

    vis.destroy_window()


def viz_loop_folder(dir, ext, out_dir):

    from glob import glob

    base_pcd = o3d.io.read_point_cloud("/home/francesco/Desktop/pcd_base_loc.ply")

    file_list = sorted(glob(dir + f"/*.{ext}"))

    vis = o3d.visualization.Visualizer()
    vis.create_window()  # width=640, height=480)

    pcd = read_asci_pc(file_list[0])
    vis.add_geometry(base_pcd)
    vis.add_geometry(pcd)
    json_fname = "/home/francesco/Desktop/o3d_render_options.json"
    vis.get_render_option().load_from_json(json_fname)
    # vis.get_render_option()
    # vis.PointColorOption =
    vis.poll_events()
    vis.update_renderer()

    for file in tqdm(file_list):
        new_pcd = read_asci_pc(file)
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        out_name = Path(file).stem
        screen_name = f"{out_dir}/{out_name}.png"
        vis.capture_screen_image(screen_name)

    vis.destroy_window()


if __name__ == "__main__":

    # Parse options from yaml file
    cfg_file = "config/config_base.yaml"
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
        images[cam] = ImageDS(cfg.paths.image_dir / cam)

    epoch_dict = {}
    for epoch in cfg.proc.epoch_to_process:
        image = Image(images[cams[0]].get_image_path(epoch))
        epoch_dict[epoch] = Path(
            (cfg.paths.results_dir)
            / f"{image.date.year}_{image.date.month:02}_{image.date.day:02}"
        ).stem

    views = [
        edict(
            {
                "front": [0.9520025, -0.24028, 0.1896109],
                "lookat": [-22.167062, 211.46727, 90.1065009791],
                "up": [-0.1559, 0.152361, 0.97594760886],
                "zoom": 0.3,
            }
        ),
        edict(
            {
                "front": [0.78205277, -0.6036, 0.15479],
                "lookat": [-22.16706245, 211.4672, 90.10650],
                "up": [-0.00041134, 0.24788848, 0.9687884851],
                "zoom": 0.4,
            }
        ),
        edict(
            {
                "front": [0.205657392, -0.956424470, 0.2072613],
                "lookat": [-22.167062, 211.4672763, 90.1065],
                "up": [0.172132, 0.243837, 0.9544],
                "zoom": 0.32,
            }
        ),
        edict(
            {
                "front": [0.85510677, -0.2782025894, 0.4374879],
                "lookat": [-22.167062, 211.4672763, 90.10650],
                "up": [-0.401234, 0.1792875, 0.8982],
                "zoom": 0.48,
            }
        ),
        edict(
            {
                "front": [0.67, -0.72, 0.17],
                "lookat": [-15.638, 222.85, 94.276],
                "up": [-0.004, 0.23, 0.972],
                "zoom": 0.8,
            }
        ),
    ]

    o3d_render_opt_fname = "res/vid/o3d_render_options.json"

    # pc_path = "/home/francesco/Desktop/test/sec_000000.asc"
    # pcd = read_asci_pc(pc_path)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.run()
    # vis.destroy_window()

    dir = "/home/francesco/Desktop/test"
    ext = "asc"
    out_dir = "/home/francesco/Desktop/test_out"
    viz_loop_folder(dir, ext, out_dir)

    # for view_id, view in enumerate(views):
    #     out_dir = Path(f"res/vid/view_{view_id}")
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     viz_loop(epoch_dict, view, o3d_render_opt_fname)

    print("Done.")
