import rasterio as rio
import numpy as np
import itertools

# import logging
# import pyproj

from pathlib import Path
from rasterio.plot import show as rshow
from rasterio.crs import CRS
from rasterio import merge as rmerge


def update_uav_dem(fname: str) -> bool:
    fname = Path(fname)
    dem = rio.open(fname)
    print(f"Dem {fname} read.")
    out, out_tform = rmerge.merge(
        [
            dem,
            dem_uav,
        ],
        method="first",
    )
    out = np.squeeze(out, 0)
    print(f"Dem {fname} merged.")

    folder = fname.parent
    fout_name = fname.name.replace("dem_", "dem_merged_")

    with rio.open(
        folder / fout_name,
        "w",
        driver="GTiff",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype="float32",
        crs=crs_rio,
        transform=out_tform,
    ) as dst:
        dst.write(out, 1)
    print(f"Merged dem saved.")

    return True


# crs_epsg = 32632
crs_proj4 = "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs +type=crs"
folder = Path("res/dem_stereo")
dem_uav_fname = "DEM_lingua_20220726_10cm_UAV.tif"

stereodem_fnames = [
    "dem_2022_05_28_10cm.tif",
    "dem_2022_06_25_10cm.tif",
    "dem_2022_07_26_10cm.tif",
    "dem_2022_08_27_10cm.tif",
]

dem_uav = rio.open(folder / dem_uav_fname)
# rshow(dem_uav)
# crs_proj4 = pyproj.CRS.from_user_input(crs_epsg).to_proj4()
crs_rio = CRS.from_proj4(crs_proj4)

fnames = [folder / f for f in stereodem_fnames]
out = list(itertools.starmap(update_uav_dem, fnames))

if all(flag == True for flag in out):
    print("Operation succeded.")


# for i, fname in enumerate(fnames):
#     if i == 0:
#         dem_uav = rio.open(folder / dem_uav_fname)
#     else:
#         prev_name = fnames[i - 1].name.replace("dem_", "dem_merged_")
#         dem_uav = rio.open(folder / prev_name)
#     if not update_uav_dem(fname):
#         print(f"Unable to merge {fname}")
#         break
