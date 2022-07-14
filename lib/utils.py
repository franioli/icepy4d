'''
MIT License

Copyright (c) 2022 Francesco Ioli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio

from rasterio.transform import Affine
from scipy.interpolate import (interp2d, 
                               griddata,
                               CloughTocher2DInterpolator,
                               LinearNDInterpolator,
                               )
from pathlib import Path 
from PIL import Image

from lib.geometry import project_points
from lib.classes import Camera
from lib.misc import create_directory

def interpolate_point_colors(pointxyz, image, camera: Camera):
    ''''
    Interpolate color of a 3D sparse point cloud, given an oriented image
      Inputs:  
        - Nx3 matrix with 3d world points coordinates
        - image as np.array in RGB channels 
            NB: if the image was impotred with OpenCV, it must be converted 
            from BGR color space to RGB
                cv2.cvtColor(image_und, cv2.COLOR_BGR2RGB)
        - Camera interior and exterior orientation matrixes: K, R, t
        - Distortion vector according to OpenCV
    Output: Nx3 colour matrix, as float numbers (normalized in [0,1])
    '''       
    # TODO: implement new checks on Camera inputs
    # assert P is not None, 'invalid projection matrix'
    assert image.ndim == 3, 'invalid input image. Image has not 3 channel'

    # import pdb; pdb.set_trace()    
    numPts = len(pointxyz)
    projections = project_points(pointxyz, camera)
    image = image.astype(np.float32) / 255.

    col = np.zeros((numPts, 3))
    for ch in range(image.shape[2]):
        col[:, ch] = bilinear_interpolate(image[:, :, ch],
                                          projections[:, 0],
                                          projections[:, 1],
                                          )
    return col


def bilinear_interpolate(im, x, y):
    ''' Perform bilinear interpolation given a 2D array (single channel image) 
    and x, y arrays of unstructured query points
    Parameters
    ----------
    im : float32
        Single channel image.
    x : float32
        nx1 array of x coordinates of query points.
    y : float32
        nx1 array of y coordinates of query points.

    Returns: nx1 array of the interpolated color
    -------
    '''
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id
    

def interpolate_point_colors_interp2d(pointxyz, image, P, K=None, dist=None, winsz=1):
    '''' Deprecated function (too slow)
    Interpolate color of a 3D sparse point cloud, given an oriented image
      Inputs:  
       - Nx3 matrix with 3d world points coordinates
       - image as np.array in RGB channels 
           NB: if the image was impotred with OpenCV, it must be converted 
           from BRG color space to RGB
               cv2.cvtColor(image_und, cv2.COLOR_BGR2RGB)
       - camera interior and exterior orientation matrixes: K, R, t
       - distortion vector according to OpenCV
    Output: Nx3 colour matrix, as float numbers (normalized in [0,1])
    '''
    # TODO: improve velocity of the function removing the cicles...
        
    assert P is not None, 'invalid projection matrix' 
    assert image.ndim == 3, 'invalid input image. Image has not 3 channel'

    if K is not None and dist is not None:
        image = cv2.undistort(image, K, dist, None, K)
    
    numPts = len(pointxyz)
    col = np.zeros((numPts,3))
    h,w,_ = image.shape
    projections = project_points(pointxyz, P, K, dist)
    image = image.astype(np.float32) / 255.
    
    for k, m in enumerate(projections):
        kint = np.round(m).astype(int)
        i = np.array([a for a in range(kint[1]-winsz,kint[1]+winsz+1)])
        j = np.array([a for a in range(kint[0]-winsz,kint[0]+winsz+1)])
        if i.min()<0 or i.max()>h or j.min()<0 or j.max()>w:
            continue
        
        
        ii, jj = np.meshgrid(i,j)
        ii, jj = ii.flatten(), jj.flatten()
        for rgb in range(0,3):
            colPatch = image[i[0]:i[-1]+1,j[0]:j[-1]+1,rgb]
            fcol = interp2d(i, j, colPatch, kind='linear')  
            col[k,rgb] = fcol(m[0], m[1])
    return col

    
#---- DSM ---##
class DSM:
    #TODO: define new better class
    ''' Class to store and manage DSM. '''
    def __init__(self, xx, yy, zz, res):
        # xx, yy = np.meshgrid(x,y)
        self.x = xx
        self.y = yy
        self.z = zz
        self.res = res    
        
def build_dsm(points3d, dsm_step=1, xlim=None, ylim=None, 
                 interp_method='linear', fill_value=np.nan,
                 save_path=None, make_dsm_plot=False
                 ):
    #TODO: Use Numpy binning instead of pandas grouping.
    
    def round_to_val(a, round_val):
        return np.round( np.array(a, dtype='float32') / round_val) * round_val

    # Check dimensions of input array
    assert np.any(np.array(points3d.shape) == 3), "Invalid size of input points"
    if points3d.shape[0] == points3d.shape[1]:
        print("Warning: input vector has just 3 points. Unable to check validity of point dimensions.")        
    if points3d.shape[0] == 3:
        points3d = points3d.T
    
    if save_path is not None:
        save_path = Path(save_path)
        save_fld = save_path.parent
        save_stem = save_path.stem
        
    # retrieve points and limits
    x, y, z = points3d[:,0], points3d[:,1], points3d[:,2]
    if xlim is None:
        xlim = [np.floor(x.min()), np.ceil(x.max())]
    if ylim is None:
        ylim = [np.floor(y.min()), np.ceil(y.max())]
        
    n_pts = len(x) 
    d_round = np.empty( [n_pts, 3] )
    d_round[:,0] = round_to_val(points3d[:,0], dsm_step)
    d_round[:,1] = round_to_val(points3d[:,1], dsm_step)
    d_round[:,2] = points3d[:,2]

    # sorting data
    ind = np.lexsort( (d_round[:,1], d_round[:,2]) )
    d_sort = d_round[ind]

    # making dataframes and grouping stuff
    df_cols = ['x_round', 'y_round', 'z']
    df = pd.DataFrame(d_sort, columns=df_cols)
    group_xy = df.groupby(['x_round', 'y_round'])
    group_mean = group_xy.mean()
    binned_df =  group_mean.index.to_frame()
    binned_df['z'] = group_mean
    
    # Move again to numpy array for interpolation
    binned_arr = binned_df.to_numpy()
    x = binned_arr[:,0].astype('float32')
    y = binned_arr[:,1].astype('float32')
    z = binned_arr[:,2].astype('float32') 
    
    # Interpolate dsm
    # import pdb; pdb.set_trace()
    xq = np.arange(xlim[0], xlim[1], dsm_step)
    yq = np.arange(ylim[0], ylim[1], dsm_step)
    grid_x, grid_y = np.meshgrid(xq,yq)
    
    if fill_value == 'mean':
        fill_value = z.mean()
    
    # dsm_grid = griddata((x, y), z, (grid_x, grid_y),
    #                     method=interp_method,
    #                     fill_value=fill_value,
    #                     )
    # interp = CloughTocher2DInterpolator(list(zip(x, y)), z)
    
    interp = LinearNDInterpolator(list(zip(x, y)), z,
                                  fill_value=fill_value,
                                  )
    dsm_grid = interp(grid_x, grid_y)

    # plot dsm 
    if make_dsm_plot:
        fig, ax = plt.subplots()
        dsm_plt = ax.contourf(grid_x, grid_y, dsm_grid)
        scatter = ax.scatter(points3d[:,0], points3d[:,1], 
                             s=5, c=points3d[:,2], 
                             marker='o',  cmap='viridis',
                             alpha=0.4, edgecolors='k',
                             )
        ax.axis('equal')
        ax.invert_yaxis()
        # fig.colorbar(dsm_plt, cax=ax, orientation='vertical')
        cbar = plt.colorbar(dsm_plt, ax=ax)
        cbar.set_label("z")
        ax.set_xlabel('x')
        ax.set_ylabel("y")
        ax.set_title('DSM interpolated from point cloud on plane X-Y')
        fig.tight_layout()
        # plt.show()
        if save_path is not None:
            plt.savefig(save_fld.joinpath(save_stem+'_plot.png'), bbox_inches='tight')

    # Save dsm as GeoTIff
    if save_path is not None:
        save_path = Path(save_path)
        create_directory(save_path.parent)
        rater_origin = [xlim[0] - dsm_step/2, ylim[0] - dsm_step/2]
        transform = Affine.translation(rater_origin[0], rater_origin[1]) \
            * Affine.scale(dsm_step, -dsm_step)
        mask = np.invert(np.isnan(dsm_grid))
        rater_origin = [grid_x[0,0], grid_y[0,0]]
        transform = Affine.translation(rater_origin[0], rater_origin[1]) \
                                       * Affine.scale(dsm_step, -dsm_step)
        with rasterio.open(
                            save_path, 'w',
                            driver='GTiff', 
                            height=dsm_grid.shape[0],
                            width=dsm_grid.shape[1], 
                            count=1,
                            dtype='float32',
                            # crs="EPSG:32632",
                            transform=transform,
                            ) as dst:
            dst.write(dsm_grid, 1)
            if fill_value is not None:
                mask = np.invert(np.isnan(dsm_grid))                                      
                dst.write_mask(mask)
            
        if fill_value is not None:
            # mask = np.invert(np.isnan(dsm_grid))
            with rasterio.open(
                                save_path.parent/(save_path.stem+'_msk.tif'), 'w',
                                driver='GTiff', 
                                height=dsm_grid.shape[0],
                                width=dsm_grid.shape[1], 
                                count=1,
                                dtype='float32',
                                # crs="EPSG:32632",
                                transform=transform,
                                ) as dst:
                dst.write(mask, 1)

    # Return a DSM object
    dsm = DSM(grid_x, grid_y, dsm_grid, dsm_step)
    
    return dsm

def generate_ortophoto(image, dsm, camera: Camera,  
                       xlim=None, ylim=None, 
                       res=None, save_path=None):
    xx = dsm.x
    yy = dsm.y
    zz = dsm.z
    if res is None:
        res = dsm.res
   
    if xlim is None:
        xlim = [xx[0,0], xx[0,-1]]
    if ylim is None:
        ylim = [yy[0,0], yy[-1,0]]
        
    dsm_shape = dsm.x.shape
    ncell = dsm_shape[0]*dsm_shape[1]
    xyz = np.zeros((ncell,3))
    xyz[:,0] = xx.flatten()
    xyz[:,1] = yy.flatten()
    xyz[:,2] = zz.flatten()
    valid_cell = np.invert(np.isnan(xyz[:,2]))
    
    cols = np.full((ncell,3), 0., 'float32')
    cols[valid_cell,:] = interpolate_point_colors(xyz[valid_cell,:], 
                                                  image, 
                                                  camera,
                                                  )
    ortophoto = np.zeros((dsm_shape[0],dsm_shape[1],3))
    ortophoto[:,:,0] = cols[:,0].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto[:,:,1] = cols[:,1].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto[:,:,2] = cols[:,2].reshape(dsm_shape[0], dsm_shape[1])
    ortophoto = np.uint8(ortophoto*255)
        
    # Save dsm as GeoTIff
    if save_path is not None:
        save_path = Path(save_path)
        create_directory(save_path.parent)
        rater_origin = [xlim[0] - res/2, ylim[0] - res/2]
        transform = Affine.translation(rater_origin[0], rater_origin[1]) \
                                       * Affine.scale(res, -res)
        with rasterio.open(
                            save_path, 'w',
                            driver='GTiff', 
                            height=ortophoto.shape[0],
                            width=ortophoto.shape[1], 
                            count=3,
                            dtype='uint8',
                            # crs="EPSG:32632",
                            transform=transform,
                            ) as dst:
            dst.write(np.moveaxis(ortophoto, -1, 0))
    
    return ortophoto

