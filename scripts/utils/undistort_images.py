import numpy as np
import logging

from pathlib import Path
from tqdm import tqdm

import src.icepy.classes as icepy_classes
import src.icepy.sfm as sfm
import src.icepy.utils.initialization as initialization
from multiprocessing import Pool, current_process

MP = True

cfg_file = "config/config_block_3_4.yaml"
cams = ["p1", "p2"]
out_folder = "res/undistored_images"

LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)


cfg = initialization.parse_yaml_cfg(cfg_file)
init = initialization.Inizialization(cfg)
init.init_image_ds()
cameras = init.init_cameras()
epoch_dict = init.init_epoch_dict()

images = {k: icepy_classes.ImageDS(Path("data/img") / k) for k in cams}
out_folder = Path(out_folder)
out_folder.mkdir(parents=True, exist_ok=True)

if MP:

    def make_zippable_lists(images, cameras, epoch_to_process):
        cam = cams[1]
        images_list = []
        cameras_list = []
        for epoch in epoch_to_process:
            images_list.append(str(images[cam].get_image_path(epoch)))
            cameras_list.append(cameras[epoch][cam])
        return images_list, cameras_list

    def undistort_image_task(image_path, camera) -> True:
        logger = logging.getLogger(current_process().name)
        img = icepy_classes.Image(image_path)
        logger.info(f"{image_path} loaded.")
        out_file = str(out_folder / f"{img.stem}_und{img.extension}")
        und = img.undistort_image(camera, out_file)
        logger.info(f"{image_path} undistored.")

        if np.any(und):
            return True
        else:
            return False

    im_list, cam_list = make_zippable_lists(images, cameras, cfg.proc.epoch_to_process)
    # undistort_image_task(im_list[0], cam_list[0])
    with Pool() as pool:
        results = pool.starmap(undistort_image_task, zip(im_list, cam_list))

else:
    cam = cams[1]
    for epoch in tqdm(cfg.proc.epoch_to_process):
        img = images[cam].read_image(epoch)
        _ = img.undistort_image(
            cameras[epoch][cam], str(out_folder / f"{img.stem}_und{img.extension}")
        )
