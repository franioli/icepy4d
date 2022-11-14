import numpy as np
from pathlib import Path
import argparse


def retrieve_sg_matches(opt):
    """ Retrieve matches saved by SuperGlue in a npz archive and save results in text files """

    npz_path = opt.npz
    outputpath = opt.output_dir

    npz_path = Path(npz_path)
    if npz_path.exists():
        try:
            npz = np.load(npz_path)
        except:
            raise IOError('Cannot load matches .npz file: %s')
    if 'npz' in locals():
        print(f'Data in "{npz_path.name}" loaded successfully')
    else:
        print('No data loaded')

    outputpath = Path(outputpath)
    if not outputpath.is_dir():
        outputpath.mkdir()

    res = {k: npz[k] for k in npz.keys()}
    for k in npz.keys():
        np.savetxt(outputpath.as_posix()+'/'+k+'.txt',
                   res[k], fmt='%.2f', delimiter=',', newline='\n')
    print('Data saved successfully')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieve SuperGlue matches from npz archive')
    parser.add_argument('--npz', metavar='<STR>',
                        type=str, help='input npz file')
    parser.add_argument('--output_dir', metavar='<STR>',
                        type=str, help='Output directory')
    opt = parser.parse_opt()

    retrieve_sg_matches(opt)
