from pathlib import Path
import numpy as np

#--- File system ---#


def create_directory(path):
    """
    Creates a directory, if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

#--- MAT ---#


def convert_to_homogeneous(x):
    '''
    Convert 2xn or 3xn vector of n points in euclidean coordinates 
    to a 3xn or 4xn vector homogeneous by adding a row of ones
    '''
    x = np.array(x)
    ndim, npts = x.shape
    if ndim != 2 and ndim != 3:
        print('Error: wrong number of dimension of the input vector.\
              A number of dimensions (rows) of 2 or 3 is required.')
        return None
    x1 = np.concatenate((x, np.ones((1, npts), 'float32')), axis=0)
    return x1


def convert_from_homogeneous(x):
    '''
    Convert 3xn or 4xn vector of n points in homogeneous coordinates 
    to a 2xn or 3xn vector in euclidean coordinates, by dividing by the 
    homogeneous part of the vector (last row) and removing one dimension
    '''
    x = np.array(x)
    ndim, npts = x.shape
    if ndim != 3 and ndim != 4:
        print('Error: wrong number of dimension of the input vector.\
              A number of dimensions (rows) of 2 or 3 is required.')
        return None
    x1 = x[:ndim-1, :] / x[ndim-1, :]
    return x1


def skew_symmetric(x):
    '''  Return skew symmetric matrix from input matrix x '''
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def compute_rmse(observed, predicted):
    ''' Compute RMSE between predicted and observed values'''
    return np.sqrt(((observed - predicted) ** 2).mean())
