import logging
import multiprocessing

from pathlib import Path
from itertools import repeat
from multiprocessing import Pool

import src.icepy.pcd_proc.meshing as meshing

MP = False

NUM_WORKERS = None

PCD_DIR = "res/point_clouds"  #
PCD_PATTERN = "dense_2022*.ply"
OUT_DIR = "res/point_clouds_meshed"


LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)


def meshing_task(pcd_path: Path, out_dir: Path, cfg: dict) -> bool:
    """
    meshing_task Task function for running meshing and sampling with a parallel pool.

    Args:
        pcd_path (Path): path to the current point cloud (is passed by map)
        out_dir (Path): output directory
        cfg (dict): configuration dictionary (see Meshing class)
    """
    m = meshing.Meshing(pcd_path, out_dir=out_dir, cfg=cfg)
    return m.run()


if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")

    cfg = {
        "do_SOR": False,
        "poisson_depth": 9,
        "min_mesh_denisty": 9,
        "save_mesh": False,
        "sample_mesh": True,
        "num_sampled_points": 4 * 10**6,
        "crop_polyline_path": "data/crop_polyline.poly",
    }

    pcd_list = sorted(Path(PCD_DIR).glob(PCD_PATTERN))
    n = len(pcd_list)

    # Test task
    # res = meshing_task(pcd_list[0], OUT_DIR, cfg)

    if MP:
        if NUM_WORKERS is None:
            p = Pool(initializer=meshing.start_process)
        else:
            p = Pool(
                processes=NUM_WORKERS,
                initializer=meshing.start_process,
            )
        result = p.starmap(
            meshing_task,
            zip(pcd_list, list(repeat(OUT_DIR, n)), list(repeat(cfg, n))),
        )
        p.close()
        p.join()

    else:
        for pcd in pcd_list:
            meshing_task(pcd, OUT_DIR, cfg)
