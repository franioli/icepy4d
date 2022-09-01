from easydict import EasyDict as edict

from lib.classes import Imageds
from lib.config import parse_yaml_cfg


def validate(cfg: edict, images: Imageds):

    cams = cfg.paths.cam_names

    # Check that number of images is the same for every camera
    for i in range(1, len(cams)):
        if len(images[cams[i]]) is not len(images[cams[i-1]]):
            raise ValueError('Error: different number of images per camera')
        else:
            print('Image datastores created successfully.')

    # Check and expand epoches to be processed
    if cfg.proc.epoch_to_process == 'all':
        cfg.proc.epoch_to_process = [x for x in range(len(images[cams[0]]))]
    if type(cfg.proc.epoch_to_process) is not list:
        raise ValueError('Invalid input of epoches to process')

    return cfg


if __name__ == '__main__':
    cfg_file = 'config/config_base.yaml'
    cfg = parse_yaml_cfg(cfg_file)

    cams = cfg.paths.cam_names

    # Create Image Datastore objects
    images = dict.fromkeys(cams)
    for cam in cams:
        images[cam] = Imageds(cfg.paths.imdir / cam)

    validate(cfg, images)
