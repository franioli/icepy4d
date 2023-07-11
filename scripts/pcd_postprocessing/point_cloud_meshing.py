import logging
import multiprocessing

from pathlib import Path
from multiprocessing import Pool

from icepy4d.point_cloud_proc.open3d_fun import MeshingPoisson

PCD_DIR = "res/point_clouds"
PCD_PATTERN = "dense_2022*.ply"  # "dense_2022_05_02.ply"  #
OUT_DIR = "res/point_clouds_meshed"

CFG = {
    "do_SOR": False,
    "poisson_depth": 9,
    "min_mesh_denisty": 9,
    "save_mesh": False,
    "sample_mesh": True,
    "num_sampled_points": 4 * 10**6,
    "crop_polyline_path": "data/crop_polyline.poly",
}


LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)

if __name__ == "__main__":

    pcd_list = sorted(Path(PCD_DIR).glob(PCD_PATTERN))
    n = len(pcd_list)
    for pcd in pcd_list:
        m = MeshingPoisson(pcd, out_dir=OUT_DIR, cfg=CFG)
        if not m.run():
            raise RuntimeError(f"Unable to mesh point cloud {pcd}")

    # Parallel processing is not effective with poisson meshing, as it is already a parallelized process.
    # multiprocessing.set_start_method("spawn")
    # NUM_WORKERS = 10
    # if NUM_WORKERS is None:
    #     p = multiprocessing.Pool()
    # else:
    #     p = multiprocessing.Pool(
    #         processes=NUM_WORKERS,
    #     )
    # result = p.starmap(
    #     meshing_task,
    #     zip(pcd_list, list(repeat(OUT_DIR, n)), list(repeat(cfg, n))),
    # )
    # p.close()
    # p.join()
