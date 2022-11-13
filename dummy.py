import yaml
from lib.config import parse_yaml_cfg
from easydict import EasyDict as edict

cfg_file = "./config/config_base.yaml"
with open(cfg_file) as file:
    yaml_opt = edict(yaml.safe_load(file))

print(yaml_opt)

yaml_opt.matching.output_dir
