import os
from pathlib import Path

try:
    import src.icepy.base_classes as icepy_classes
    import src.icepy.sfm as sfm
    import src.icepy.utils.initialization as initialization
except:
    print("Unable to import Icepy package")
    exit

cfg_file = "config/config_block_3_4.yaml"
cfg_file = "assets/config.yaml"

cfg = initialization.parse_yaml_cfg(cfg_file)

cams = ["p1", "p2"]
images = {k: icepy_classes.ImageDS(Path("data/img") / k) for k in cams}

init = initialization.Inizialization(cfg)
init.init_image_ds()
epoch_dict = init.init_epoch_dict()
