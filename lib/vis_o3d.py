import numpy as np
import open3d as o3d

from pathlib import Path
from time import sleep
from easydict import EasyDict as edict
from tqdm import tqdm

from lib.base_classes.images import Image, Imageds
from lib.read_config import parse_yaml_cfg


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


def main(epoch_dict, view, o3d_render_opt_fname):

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

        out_dir = Path(f"res/vid/view_{view_id}")
        out_dir.mkdir(parents=True, exist_ok=True)

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
        images[cam] = Imageds(cfg.paths.image_dir / cam)

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

    for view_id, view in enumerate(views):
        main(epoch_dict, view, o3d_render_opt_fname)

    print("Done.")
