import numpy as np
import open3d as o3d

from typing import List
from pathlib import Path
from easydict import EasyDict as edict
from tqdm import tqdm

import icepy4d.point_cloud_proc.open3d_fun as o3d_fun
from icepy4d.utils.transformations import belvedere_utm2loc


def viz_sections_loop(
    pcd_list: List[str],
    out_dir: str,
    o3d_render_opt_fname: str,
    base_pcd_fname: str,
    polyline_fname: str,
    pcd_color: List[float] = [1.0, 0.0, 0.0],
    window_size: List[int] = [1920, 1080],
    hide_window: bool = False,
):
    base_pcd = o3d.io.read_point_cloud(str(base_pcd_fname))
    pcd = o3d.io.read_point_cloud(str(pcd_list)[0])
    pcd = o3d_fun.filter_pcd_by_polyline(pcd, polyline_fname)
    pcd.paint_uniform_color(pcd_color)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=window_size[0],
        height=window_size[1],
        left=200,
        top=50,
        visible=not hide_window,
    )
    vis.add_geometry(base_pcd)
    vis.add_geometry(pcd)

    vis.get_render_option().load_from_json(o3d_render_opt_fname)
    vis.poll_events()
    vis.update_renderer()

    for file in tqdm(pcd_list):
        new_pcd = o3d.io.read_point_cloud(str(file))
        new_pcd = o3d_fun.filter_pcd_by_polyline(new_pcd, polyline_fname)
        new_pcd.paint_uniform_color(pcd_color)

        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        out_name = Path(file).stem.replace("dense_", "")
        screen_name = f"{out_dir}/{out_name}.png"
        vis.capture_screen_image(screen_name, do_render=True)

        del new_pcd

    vis.destroy_window()


if __name__ == "__main__":
    pcd_dir = "res/point_clouds"
    pcd_name_pattern = "dense_*.ply"
    polyline_fname = "scripts/dynamic_visualization/sections/sec_bbox.poly"
    o3d_render_opt_fname = "scripts/dynamic_visualization/o3d_render_options.json"
    base_pcd_fname = (
        "scripts/dynamic_visualization/sections/pcd_base_loc_translatedZ.ply"
    )
    out_dir = "scripts/dynamic_visualization/sections/frames"

    pcd_dir = Path(pcd_dir)
    pcd_list = sorted(pcd_dir.glob(pcd_name_pattern))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transform base pcd to local coordinates
    # base_pcd = o3d.io.read_point_cloud(str(base_pcd_fname))
    # T = belvedere_utm2loc()
    # base_pcd.transform(T)
    # o3d.io.write_point_cloud(str(base_pcd_fname), base_pcd)

    viz_sections_loop(
        pcd_list,
        out_dir,
        base_pcd_fname=base_pcd_fname,
        polyline_fname=polyline_fname,
        o3d_render_opt_fname=o3d_render_opt_fname,
        window_size=[2 * 1920, 2 * 1080],
        hide_window=False,
    )

    print("Done.")
