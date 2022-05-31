import numpy as np
import cv2 
import torch
import argparse
import time
import os, glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# class Camera:
#     """ Class to help manage Cameras. """
    
#     """Initialise the camera calibration object 
#     Parameters
#     ----------
#     *args : str
#       Read calibration text file, with Full OpenCV parameters in a row.
#     """

#     def __init__(self, *args):
        
#         self.reset()
#         if args[0] == None:
#             print('No calibration file available. Setting default parameters')
#         self.K = np.array([[6620.56653699, 0., 3020.0385], [0., 6619.88882, 1886.01352], [0., 0., 1.]] )
        
    
#     def reset(self):
#         self.K = np.array([[500, 0., 500], [0., 500, 500], [0., 0., 1.]] )
#         self.R = np.identity(3)
#         self.t = np.zeros((3,))
#         self.P = cv2.projectionFromKRt(self.K, self.R, self.t)
        
               
# class images:
#     """ Class to help manage Cameras. """
    
#     """Initialise the camera calibration object 
#     Parameters
#     ----------
#     *args : str
#       Read calibration text file, with Full OpenCV parameters in a row.
#     """

#     def __init__(self, *args):
        
#         self.reset()

        
#     def reset(self):
        
# --- PREPROCESSING ---
def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, color, resize, crop, resize_float):
    if color:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)
    
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if crop:
       image = image[ crop[1]:crop[3],crop[0]:crop[2] ]

    inp = frame2tensor(image, device)
    return image, inp, scales


def read_img(path, color, resize, crop):
    if color:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)
    
    if image is None:
        return None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    image = cv2.resize(image, (w_new, h_new))
    
    if crop:
       image = image[ crop[1]:crop[3],crop[0]:crop[2] ]

    return image, scales

def generateTiles(image, rowDivisor=2, colDivisor=2, overlap = 200, viz=False, out_dir='tiles', writeTile2Disk=True):
    assert not (image is None), 'Invalid image input'
        
    image = image.astype('float32')
    H = image.shape[0]
    W = image.shape[1]    
    DY  = round(H/rowDivisor/10)*10;
    DX  = round(W/colDivisor/10)*10;
    dim = (rowDivisor,colDivisor)
    
    # Check image dimensions
    # if not W % colDivisor == 0:
    #     print('Number of columns non divisible by the ColDivisor. Removing last column.')
    #     image = image[:, 0:-1]
    # if not H % rowDivisor == 0:
    #     print('Number of rows non divisible by the RowDivisor. Removing last row')
    #     image = image[0:-1, :]
    
    tiles = [] 
    limits = []   
    for col in range(0, colDivisor):
        for row in range(0, rowDivisor):
            tileIdx = np.ravel_multi_index((row,col), dim, order='F');
            limits.append((max(0, col*DX - overlap), 
                           max(0, row*DY - overlap),
                           max(0, col*DX - overlap) + DX+overlap,
                           max(0, row*DY - overlap) + DY+overlap ) )
            # print(f'Tile {tileIdx}: xlim = ({ limits[tileIdx][0], limits[tileIdx][2]}), ylim = {limits[tileIdx][1], limits[tileIdx][3]}')
            tile = image[limits[tileIdx][1]:limits[tileIdx][3], 
                         limits[tileIdx][0]:limits[tileIdx][2] ]
            tiles.append(tile)
            if writeTile2Disk:
                isExist = os.path.exists(out_dir)
                if not isExist:
                    os.makedirs(out_dir)       
                cv2.imwrite(os.path.join(out_dir,'tile_'+str(tileIdx)+'_'
                         +str(limits[tileIdx][0])+'_'+str(limits[tileIdx][1])+'.jpg'), tile)

    return tiles, limits

    
# --- TIMING ---
class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()  
               