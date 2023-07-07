import numpy as np
import open3d as o3d
from pathlib import Path
from icepy4d.utils.transformations import Rotrotranslation


PCD_DIR = "res/monthly_pcd/"
PCD_PATTERN = "*"

pcd_dir = (Path.cwd() / Path(PCD_DIR)).resolve()
pcd_list = sorted(pcd_dir.glob("*"))
n = len(pcd_list)

pcd_name = pcd_list[0]


def loc2utm(pcd_path: str, out_path: str) -> bool:
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pcd.points)

    belv_rotra = Rotrotranslation(Rotrotranslation.belvedere_loc2utm())
    pts_utm = belv_rotra.apply_transformation(pts.T).T

    pcd.points = o3d.utility.Vector3dVector(pts_utm)
    o3d.io.write_point_cloud(str(out_path), pcd)
    return True


def utm2loc(pcd_path: str, out_path: str) -> bool:
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pcd.points)

    belv_rotra = Rotrotranslation(Rotrotranslation.belvedere_utm2loc())
    pts_utm = belv_rotra.apply_transformation(pts.T).T

    pcd.points = o3d.utility.Vector3dVector(pts_utm)
    o3d.io.write_point_cloud(str(out_path), pcd)
    return True


# convert pcd to utm
for pcd_name in pcd_list:
    loc2utm(pcd_name, pcd_dir / f"{pcd_name.stem}_utm.ply")


pcd_name = "res/monthly_pcd/belvedere2021_densaMedium_lingua_50cm_utm.ply"
utm2loc(pcd_name, pcd_dir / f"{Path(pcd_name).stem}_loc.ply")
