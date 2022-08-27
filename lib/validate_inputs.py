from classes import Imageds
from easydict import EasyDict as edict
from config import parse_yaml


def validate(cfg: edict, images: Imageds):

    cams = cfg.paths.cam_names

    # Check that number of images is the same for every camera
    for i in range(1, len(cams)):
        if len(images[cams[i]]) is not len(images[cams[i-1]]):
            print('Error: different number of images per camera')
            raise SystemExit(0)
        else:
            print('Image datastores created successfully.')

    # Check and expand epoches to be processed
    if cfg.proc.epoch_to_process == 'all':
        cfg.proc.epoch_to_process = [x for x in range(len(images[cams[0]]))]
    if type(cfg.proc.epoch_to_process) is not list:
        print('Invalid input of epoches to process')
        raise SystemExit(0)

    return cfg


if __name__ == '__main__':
    cfg_file = 'config/config_base.yaml'
    cfg = parse_yaml(cfg_file)

    cams = cfg.paths.cam_names
    images = dict.fromkeys(cams)

    # Create Image Datastore objects
    for cam in cams:
        images[cam] = Imageds(cfg.paths.imdir / cam)

    validate(cfg, images)
