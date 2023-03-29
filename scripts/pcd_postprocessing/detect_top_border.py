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

PCD_DIR = "res/detect_top_border/"
PCD_PATTERN = "border_*.ply"
POLYLINE_PATH = "res/detect_top_border/poly.poly"
OUT_DIR = "res/detect_top_border"
FOUT_NAME = "top_border_coords.txt"

LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)


def merge_sparse_dense():

    PCD_DIR = "res/point_clouds"
    PCD_PATTERN = "dense_2022*.ply"

    pcd_list = sorted(Path(PCD_DIR).glob(PCD_PATTERN))
    output_dir = Path(OUT_DIR)
    output_dir.mkdir(exist_ok=True)

    for pcd_path in pcd_list:
        logging.info(f"Processing pcd {pcd_path.name}")

        fnames = [str(pcd_path), str(pcd_path).replace("dense", "sparse")]

        out_name = pcd_path.name.replace("dense", "merged")
        pcd = read_and_merge_point_clouds(fnames)
        cropped = filter_pcd_by_polyline(pcd, POLYLINE_PATH, dir="x")
        o3d.io.write_point_cloud(str(output_dir / out_name), cropped)
        # o3d.visualization.draw_geometries([cropped])


def export_coordinates_as_scalar_fields(pcd) -> True:
    sf_names = ["Coord. X", "Coord. Y", "Coord. Z"]
    coords = pcd.toNpArrayCopy()
    for i, name in enumerate(sf_names):
        sf_num = pcd.addScalarField(name)
        sf = pcd.getScalarField(sf_num)
        sf.fromNpArrayCopy(coords[:, i])
    return True


def detect_border_by_geometry():
    PCD_PATTERN = "merged_*.ply"
    pcd_list = sorted(Path(PCD_DIR).glob(PCD_PATTERN))
    output_dir = Path(OUT_DIR)
    output_dir.mkdir(exist_ok=True)
    for pcd_path in pcd_list:
        base_name = PCD_PATTERN.split("_")[0]
        pcd_date = pcd_path.stem.replace(base_name + "_", "")
        logging.info(f"Processing pcd {pcd_date}")

        pcd = cc.loadPointCloud(str(pcd_path))
        if pcd is None:
            raise IOError(f"Unable to read point cloud {pcd_path}")

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
        if not export_coordinates_as_scalar_fields(pcd_filtered):
            raise RuntimeError(f"Unable to export coordinates as scalar")

        # path = str(output_dir / ("filtered_" + pcd_date + ".ply"))
        # if not cc.SavePointCloud(pcd_filtered, path):
        #     raise IOError(f"Unable to save cropped point cloud to {path}.")

        lim_percentile = (60, 95)
        z_coord_sf = pcd_filtered.toNpArrayCopy()[:, 2]
        sf_num = pcd_filtered.getScalarFieldDic()["Coord. Z"]
        sf_min = np.nanpercentile(z_coord_sf, lim_percentile[0])
        sf_max = np.nanpercentile(z_coord_sf, lim_percentile[1])
        pcd_filtered.setCurrentOutScalarField(sf_num)
        pcd_border = cc.filterBySFValue(sf_min, sf_max, pcd_filtered)

        path = str(output_dir / ("border_" + pcd_date + ".ply"))
        if not cc.SavePointCloud(pcd_border, path):
            raise IOError(f"Unable to save cropped point cloud to {path}.")

        cc.deleteEntity(pcd)
        cc.deleteEntity(pcd_filtered)


def extract_glacier_border():

    pcd_list = sorted(Path(PCD_DIR).glob(PCD_PATTERN))
    output_dir = Path(OUT_DIR)
    output_dir.mkdir(exist_ok=True)
    base_name = PCD_PATTERN.split("_")[0]

    logging.info("Computing border median coordinates...")
    with open(output_dir / FOUT_NAME, "w") as f:
        f.write(
            f"pcd_name,date,x_mean,x_median,x_std,y_mean,y_median,y_std,z_mean,z_median,z_std\n"
        )
        for pcd_path in pcd_list:
            logging.info(f"Processing pcd {pcd_path.name}")
            try:
                pcd_date = pcd_path.stem.replace(base_name + "_", "")
                pcd_border = cc.loadPointCloud(str(pcd_path))
                if pcd_border is None:
                    raise IOError(f"Unable to read point cloud {pcd_path}")

                ylims = (222.0, 228.0)
                try:
                    sf_num = pcd_border.getScalarFieldDic()["Coord. Y"]
                except:
                    raise RuntimeError("Unable to get Coord. Y scalar field")
                pcd_border.setCurrentOutScalarField(sf_num)
                pcd_border = cc.filterBySFValue(ylims[0], ylims[1], pcd_border)

                x_coord_sf = pcd_border.toNpArrayCopy()[:, 0]
                median_x = np.median(x_coord_sf)
                sf_min = median_x - 10
                sf_max = median_x + 10
                try:
                    sf_num = pcd_border.getScalarFieldDic()["Coord. X"]
                except:
                    raise RuntimeError("Unable to get Coord. X scalar field")
                pcd_border.setCurrentOutScalarField(sf_num)
                pcd_border = cc.filterBySFValue(sf_min, sf_max, pcd_border)

                coords = pcd_border.toNpArrayCopy()
                mean = np.mean(coords, axis=0)
                median = np.median(coords, axis=0)
                std = np.std(coords, axis=0)

                f.write(
                    f"{pcd_path.name},{pcd_date},{mean[0]:.3f},{median[0]:.3f},{std[0]:.3f},{mean[1]:.3f},{median[1]:.3f},{std[1]:.3f},{mean[2]:.3f},{median[2]:.3f},{std[2]:.3f}\n"
                )
            except RuntimeError as err:
                logging.exception(err)
                f.write(
                    f"{pcd_path.name},{pcd_date},NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN\n"
                )
                logging.info("Moving to next pcd.")
            cc.deleteEntity(pcd_border)


if __name__ == "__main__":

    # detect_border_by_geometry()
    extract_glacier_border()

    print("done.")
