import logging
import numpy as np
import open3d as o3d

from pathlib import Path
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


if __name__ == "__main__":

    detect_glacier_border()

    print("done.")
