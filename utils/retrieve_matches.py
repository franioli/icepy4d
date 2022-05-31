import numpy as np
from pathlib import Path
import argparse

if __name__ == '__main__':
    """ Retrieve matches saved by SuperGlue in a npz archive and save results in text files """

    parser = argparse.ArgumentParser(description='Retrieve SuperGlue matches from npz archive')
    parser.add_argument('--npz', metavar='<STR>', type=str, help='input npz file')
    parser.add_argument('--output_dir', metavar='<STR>', type=str, help='Output directory')
    args = parser.parse_args()

    npzpath = args.npz
    outputpath = args.output_dir
        
    npzpath = Path(npzpath)
    if npzpath.exists():
        try:
            npz = np.load(npzpath)
        except:
            raise IOError('Cannot load matches .npz file: %s')
    if 'npz' in locals():
        print(f'Data in "{npzpath.name}" loaded successfully' )
    else: 
        print('No data loaded')
    
    outputpath = Path(outputpath)
    if not outputpath.is_dir():
        outputpath.mkdir()    
    
    # keys = ['keypoints0', 'keypoints1', 'matches0', 'match_confidence', 'mkpts0', 'mkpts1', 'kpts0_fullres', 'kpts1_fullres', 'mkpts0_fullres', 'mkpts1_fullres']
    res = {k: npz[k] for k in npz.keys()}
    for k in npz.keys():
        np.savetxt(outputpath.as_posix()+'/'+k+'.txt',  res[k], fmt='%.2f', delimiter=',', newline='\n')
    print('Data saved successfully')
    
