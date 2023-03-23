import logging
import numpy as np
import open3d as o3d

from pathlib import Path
from typing import Union, List
from tqdm import tqdm

import cloudComPy as cc


from icepy.point_cloud_proc.open3d_fun import (
    filter_pcd_by_polyline,
    read_and_merge_point_clouds,
)

PCD_DIR = "res/point_clouds"
PCD_PATTERN = "dense_2022*.ply"
OUT_DIR = "res/point_clouds_meshed"

LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)


def detect_glacier_border():

    pcd_list = sorted(Path(PCD_DIR).glob(PCD_PATTERN))
    n = len(pcd_list)

    polyline_path = "test/poly.poly"
    output_dir = "test/detect_border"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    pcd_path = pcd_list[0]

    with open(output_dir / "surf_heigth.txt", "w") as f:
        f.write(f"pcd_name,mean,median,std,min,max\n")

        for pcd_path in pcd_list:
            logging.info(f"Processing pcd {pcd_path.name}")

            fnames = [str(pcd_path), str(pcd_path).replace("dense", "sparse")]

            out_name = pcd_path.name.replace("dense", "merged")
            pcd = read_and_merge_point_clouds(fnames)
            cropped = filter_pcd_by_polyline(pcd, polyline_path, dir="x")
            o3d.io.write_point_cloud(str(output_dir / out_name), cropped)
            # o3d.visualization.draw_geometries([cropped])

            path = str(output_dir / out_name)
            pcd = cc.loadPointCloud(path)
            if pcd is None:
                raise IOError(f"Unable to read point cloud {path}")

            # if not cc.computeFeature( cc.GeomFeature.Linearity, 2, [pcd]):
            #     raise RuntimeError(
            #         f"Unable to compute geometric features from point cloud {path}"
            #     )
            # logging.info("Geometric features computed successfully.")

            # radius = 2.0
            # feature_to_compute = cc.GeomFeature.Linearity
            # lim_percentile = (95, 100)
            # featstr = str(feature_to_compute).split(".")[-1]

            radius_list = [2.0, 0.15]
            features_to_compute = [
                cc.GeomFeature.Linearity,
                cc.GeomFeature.Verticality,
            ]

            logging.info(f"Computing geometric features")
            for feature, radius in tqdm(zip(features_to_compute, radius_list)):
                if not cc.computeFeature(feature, radius, [pcd]):
                    raise RuntimeError(
                        f"Unable to compute geometric features from point cloud {path}"
                    )
            logging.info("Geometric features computed successfully.")

            def export_coordinates_as_scalar_fields(pcd) -> True:
                sf_names = ["Coord. X", "Coord. Y", "Coord. Z"]
                coords = pcd.toNpArrayCopy()
                for i, name in enumerate(sf_names):
                    sf_num = pcd.addScalarField(name)
                    sf = pcd.getScalarField(sf_num)
                    sf.fromNpArrayCopy(coords[:, i])
                return True

            # Make Linearity sf active and filter by percentile
            lim_percentile = (95, 100)
            dic = pcd.getScalarFieldDic()
            sf_num = dic[list(dic.keys())[-2]]
            sf = pcd.getScalarField(sf_num)
            pcd.setCurrentOutScalarField(sf_num)
            sf_min = np.nanpercentile(sf.toNpArray(), lim_percentile[0])
            sf_max = np.nanpercentile(sf.toNpArray(), lim_percentile[1])
            pcd_filtered = cc.filterBySFValue(sf_min, sf_max, pcd)

            # Make Verticality sf active and filter by percentile
            lim_percentile = (95, 100)
            dic = pcd_filtered.getScalarFieldDic()
            sf_num = dic[list(dic.keys())[-1]]
            sf = pcd_filtered.getScalarField(sf_num)
            pcd_filtered.setCurrentOutScalarField(sf_num)
            sf_min = np.nanpercentile(sf.toNpArray(), lim_percentile[0])
            sf_max = np.nanpercentile(sf.toNpArray(), lim_percentile[1])
            pcd_filtered = cc.filterBySFValue(sf_min, sf_max, pcd_filtered)
            export_coordinates_as_scalar_fields(pcd_filtered)

            path = str(output_dir / out_name.replace("merged", "filtered"))
            if not cc.SavePointCloud(pcd_filtered, path):
                raise IOError(f"Unable to save cropped point cloud to {path}.")

            lim_percentile = (60, 95)
            z_coord_sf = pcd_filtered.toNpArrayCopy()[:, 2]
            sf_num = pcd_filtered.getScalarFieldDic()["Coord. Z"]
            sf_min = np.nanpercentile(z_coord_sf, lim_percentile[0])
            sf_max = np.nanpercentile(z_coord_sf, lim_percentile[1])
            pcd_filtered.setCurrentOutScalarField(sf_num)
            pcd_border = cc.filterBySFValue(sf_min, sf_max, pcd_filtered)

            path = str(output_dir / out_name.replace("merged", "border"))
            if not cc.SavePointCloud(pcd_border, path):
                raise IOError(f"Unable to save cropped point cloud to {path}.")

            ylims = (220.0, 230.0)
            sf_num = pcd_border.getScalarFieldDic()["Coord. Y"]
            pcd_border.setCurrentOutScalarField(sf_num)
            pcd_border = cc.filterBySFValue(ylims[0], ylims[1], pcd_border)

            x_coord_sf = pcd_border.toNpArrayCopy()[:, 0]
            median_x = np.median(x_coord_sf)
            sf_min = median_x - 20
            sf_max = median_x + 20
            sf_num = pcd_border.getScalarFieldDic()["Coord. X"]
            pcd_border.setCurrentOutScalarField(sf_num)
            pcd_border = cc.filterBySFValue(sf_min, sf_max, pcd_border)

            path = str(output_dir / out_name.replace("merged", "border_cut"))
            if not cc.SavePointCloud(pcd_border, path):
                raise IOError(f"Unable to save cropped point cloud to {path}.")

            z_coord = pcd_border.toNpArrayCopy()[:, 2]
            mean_z = np.mean(z_coord)
            median_z = np.median(z_coord)
            std_z = np.std(z_coord)
            min_z = np.min(z_coord)
            max_z = np.max(z_coord)

            f.write(
                f"{pcd_path.name},{mean_z:.3f},{median_z:.3f},{std_z:.3f},{min_z:.3f},{max_z:.3f}\n"
            )

            cc.deleteEntity(pcd)
            cc.deleteEntity(pcd_filtered)
            cc.deleteEntity(pcd_border)


def voxelize():

    import open3d as o3d
    from tqdm import tqdm

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

    # detect_glacier_border()
    voxelize()

    print("done.")
