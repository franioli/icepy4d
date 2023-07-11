import logging
import multiprocessing
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import open3d as o3d
from icepy4d.post_processing.open3d_fun import MeshingPoisson
from icepy4d.utils.dsm_orthophoto import build_dsm
from icepy4d.utils.transformations import Rotrotranslation, belvedere_loc2utm

PCD_DIR = "res/monthly_pcd/"
PCD_PATTERN = "*"


LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)

pcd_dir = (Path.cwd() / Path(PCD_DIR)).resolve()
pcd_list = sorted(pcd_dir.glob("*"))
n = len(pcd_list)

pcd_name = pcd_list[0]

# convert pcd to utm
for pcd_name in pcd_list:
    pcd = o3d.io.read_point_cloud(str(pcd_name))
    pts = np.asarray(pcd.points)

    belv_rotra = Rotrotranslation(belvedere_loc2utm())
    pts_utm = belv_rotra.apply_transformation(pts.T).T

    pcd.points = o3d.utility.Vector3dVector(pts_utm)
    o3d.io.write_point_cloud(str(pcd_dir / f"{pcd_name.stem}_utm.ply"), pcd)

# # build dsm
# dsm_out = pcd_dir / f"{pcd.stem}.tif"
# dem = build_dsm(pts_utm, dsm_step=0.1, save_path=dsm_out)


print("done")
