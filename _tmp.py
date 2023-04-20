# %%
from pathlib import Path
import open3d as o3d

from icepy4d.classes import PointCloud

import cloudComPy as cc  # import the CloudComPy module

from multiprocessing import Pool

STEP = 10
root_path = Path.cwd()
path = "res/point_clouds"
pattern = "dense_*.ply"
dest_dir = Path("pcd_potree")


dest_dir.mkdir(exist_ok=True)
path = Path(root_path) / path

files = sorted(Path(path).glob(f"{pattern}"))
for i, file in enumerate(files):
    if i % STEP != 0:
        continue
    print(f"Processing {file}")
    pcd = cc.loadPointCloud(str(file))
    ret = cc.SavePointCloud(pcd, str(dest_dir / f"{file.stem}.las"))
    cc.deleteEntity(pcd)
