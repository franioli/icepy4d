import logging
import numpy as np
import open3d as o3d

from pathlib import Path
from typing import Union, List
from tqdm import tqdm

PCD_DIR = "res/point_clouds"
PCD_PATTERN = "dense_2022*.ply"
OUT_DIR = "res/point_clouds_meshed"

LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)


def voxelize():

    PCD_DIR = "test/voxelization"

    pcd_list = sorted(Path(PCD_DIR).glob(PCD_PATTERN))
    n = len(pcd_list)

    pcd_path = pcd_list[0]
    for pcd_path in pcd_list:
        print(f"Voxelization of pcd {pcd_path.stem}")
        pcd = o3d.io.read_point_cloud(str(pcd_path))

        voxel_size = 0.2
        bb_min = [-100, 130, 60]
        bb_max = [30, 330, 120]
        bb_min = np.array(bb_min).astype(np.float64)
        bb_max = np.array(bb_max).astype(np.float64)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd, voxel_size, bb_min, bb_max
        )
        voxel_grid.get_axis_aligned_bounding_box()

        fout = str(Path(PCD_DIR) / "voxel_grid.ply")
        o3d.io.write_voxel_grid(
            fout, voxel_grid, write_ascii=True, compressed=False, print_progress=False
        )

        # Save only filled voxels to file
        voxels = voxel_grid.get_voxels()
        fout = Path(PCD_DIR) / f"{pcd_path.stem}_voxel_{voxel_size}m.txt"
        with open(fout, "w") as f:
            for v in tqdm(voxels):
                x, y, z = voxel_grid.get_voxel_center_coordinate(v.grid_index)
                r, g, b = v.color
                f.write(f"{x:.4f},{y:.4f},{z:.4f},{r},{g},{b}\n")

        # # Save all voxels and also empty voxels with nan
        # from itertools import product

        # x = np.arange(bb_min[0], bb_max[0], voxel_size)
        # y = np.arange(bb_min[1], bb_max[1], voxel_size)
        # z = np.arange(bb_min[2], bb_max[2], voxel_size)

        # voxels = voxel_grid.get_voxels()
        # fout = Path(PCD_DIR) / f"{pcd_path.stem}_voxel_with_nan.txt"
        # logging.info("Saving started...")
        # with open(fout, "w") as f:
        #     voxel_indices = np.array([v.grid_index for v in voxels])
        #     voxel_colors = np.array([v.color for v in voxels])
        #     voxel_centers = np.array(
        #         [
        #             voxel_grid.get_voxel_center_coordinate(v_idx)
        #             for v_idx in voxel_indices
        #         ]
        #     )

        #     node_indices = np.array(list(product(x, y, z))).astype(np.int32)
        #     node_mask = np.all(np.isin(node_indices, voxel_indices), axis=1)

        #     nan_rows = np.full((len(node_indices),), "NaN")
        #     data_rows = np.full((len(node_indices), 6), np.nan)
        #     data_rows[node_mask, :3] = voxel_centers
        #     data_rows[node_mask, 3:] = voxel_colors[np.where(node_mask)[0], :]

        #     rows = np.where(node_mask, data_rows, nan_rows)
        #     np.savetxt(f, rows, fmt="%.4f,%d,%d,%d", delimiter=",", newline="\n")

        # from itertools import product
        # from multiprocessing import Pool

        # x = np.arange(bb_min[0], bb_max[0], voxel_size)
        # y = np.arange(bb_min[1], bb_max[1], voxel_size)
        # z = np.arange(bb_min[2], bb_max[2], voxel_size)

        # voxels = voxel_grid.get_voxels()
        # fout = Path(PCD_DIR) / f"{pcd_path.stem}_voxel_with_nan.txt"

        # logging.info("Saving started...")
        # filled_voxels = [tuple(v.grid_index) for v in voxels]

        # # def task(node):
        # #     vox_idx = tuple(np.array(node).astype(np.int32))
        # #     if vox_idx in filled_voxels:
        # #         id = [i for i, v in enumerate(voxels) if v.grid_index == vox_idx][0]
        # #         xyz = voxel_grid.get_voxel_center_coordinate(vox_idx)
        # #         rgb = voxels[id].color
        # #         out = (xyz, rgb)
        # #     else:
        # #         x, y, z = node
        # #         rgb = ["NaN", "NaN", "NaN"]
        # #         out = (xyz, rgb)
        # #     return out

        # # with Pool(processes=10) as pool:
        # #     result = pool.map(task, product(x, y, z))

        # with open(fout, "w") as f:
        #     for node in tqdm(product(x, y, z)):
        #         vox_idx = tuple(np.array(node).astype(np.int32))
        #         if vox_idx in filled_voxels:
        #             id = [i for i, v in enumerate(voxels) if v.grid_index == vox_idx][0]
        #             x, y, z = voxel_grid.get_voxel_center_coordinate(vox_idx)
        #             r, g, b = voxels[id].color
        #             f.write(f"{x:.4f},{y:.4f},{z:.4f},{r},{g},{b}\n")
        #         else:
        #             x, y, z = node
        #             f.write(f"{x:.4f},{y:.4f},{z:.4f},NaN,NaN,NaN\n")

        # for node in product(x, y, z):
        #     for i, vox in enumerate(voxels):
        #         if np.allclose(np.array(node).astype(np.int32), vox.grid_index):
        #             vox_idx = vox.grid_index
        #             id = i
        #         else:
        #             vox_idx = -1
        #     if vox_idx > -1:
        #         x, y, z = voxel_grid.get_voxel_center_coordinate(vox_idx)
        #         r, g, b = voxels[id].color
        #         f.write(f"{x:.4f},{y:.4f},{z:.4f},{r},{g},{b}\n")
        #     else:
        #         f.write("NaN\n")

        logging.info("Saving completed.")

        print("done.")

    o3d.visualization.draw_geometries([voxel_grid])

    vox_mesh = o3d.geometry.TriangleMesh()
    for v in voxels:
        cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube.paint_uniform_color(v.color)
        cube.translate(v.grid_index, relative=False)
        vox_mesh += cube
    vox_mesh.translate([0.5, 0.5, 0.5], relative=True)
    vox_mesh.scale(voxel_size, [0, 0, 0])
    vox_mesh.translate(voxel_grid.origin, relative=True)
    vox_mesh.merge_close_vertices(0.0000001)
    fout = str(Path(PCD_DIR) / "vox_mesh.ply")
    o3d.io.write_triangle_mesh(fout, vox_mesh)


if __name__ == "__main__":

    voxelize()

    print("done.")
