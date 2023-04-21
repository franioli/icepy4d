import numpy as np
from lmfit import Minimizer, Parameters, fit_report
from scipy import stats

from icepy4d.thirdparty.transformations import euler_matrix
from icepy4d.utils.math import convert_from_homogeneous, convert_to_homogeneous


def get_T_from_params(
    params: Parameters,
):
    # Get parameters
    parvals = params.valuesdict()
    rx = parvals["rx"]
    ry = parvals["ry"]
    rz = parvals["rz"]
    tx = parvals["tx"]
    ty = parvals["ty"]
    tz = parvals["tz"]
    m = parvals["m"]

    # Build 4x4 transformation matrix (T) in homogeneous coordinates
    T = np.identity(4)
    R = euler_matrix(rx, ry, rz)
    T[0:3, 0:3] = (m * np.identity(3)) @ R[:3, :3]
    T[0:3, 3:4] = np.array([tx, ty, tz]).reshape(3, 1)

    return T


def compute_residuals(
    params: Parameters,
    x0: np.ndarray,
    x1: np.ndarray,
    weights: np.ndarray = None,
    prior_covariance_scale: float = None,
) -> np.ndarray:
    """3D rototranslation with scale factor

    X1_ = T_ + m * R * X0_

    Inputs:
    - x0 (np.ndarray): Points in the starting reference system
    - x1 (np.ndarray): Points in final reference system
    - weights (np.ndarray, defult = None): weights (e.g., inverse of a-priori observation uncertainty)
    - prior_covariance_scale (float, default = None): A-priori sigma_0^2

    Return:
    - res (nx1 np.ndarray): Vector of the weighted residuals

    """

    # Get parameters from params
    T = get_T_from_params(params)

    # Convert points to homogeneos coordinates and traspose np array to obtain a 4xn matrix
    # Note that here the function convert_to_homogeneous is different than that implemented in rotra3d repo and it requires the dimensions to be along the colums direction.
    x0 = convert_to_homogeneous(x0.T)

    # Apply transformation to x0 points
    x1_ = T @ x0
    x1_ = x1_[:3, :].T

    # Compute residuals as differences between observed and estimated values, scaled by the a-priori observation uncertainties
    res = x1 - x1_

    # If weigthts are provided, scale residual
    if weights is not None:
        if weights.shape != res.shape:
            raise ValueError(
                f"Wrong dimensions of the weight matrix. It must be of size {res.shape}"
            )

        res = res * weights

    return res.flatten()


def rescale_residuals(
    residuals: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    residuals = residuals / weights
    return residuals


def print_results(
    result,
    weights: np.ndarray = None,
    sigma0_2: float = 1.0,
) -> None:
    ndim = weights.shape[1]

    # Rescale residuals
    if weights is not None:
        residuals = rescale_residuals(
            result.residual.reshape(-1, ndim),
            weights,
        )
    else:
        residuals = result.residual.reshape(-1, ndim)

    print("-------------------------------")
    print("Optimization report")
    print(fit_report(result))

    print("-------------------------------")
    print(f"Chi quadro test:")
    nfree = result.nfree
    chi_lim = stats.chi2.ppf(1 - 0.05, df=nfree)
    chi_0 = result.redchi / sigma0_2
    print(f"Degrees of freedom: {nfree}")
    print(f"Chi2 empirical: {chi_0:.3f}")
    print(f"Chi2 limit: {chi_lim:.3f}")
    if chi_0 < chi_lim:
        print("Test passed")
    else:
        print("Test NOT passed")

    print("-------------------------------")
    print("Residuals")
    # print('     X       Y      Z')
    # print(f'{res[0]:8.3f} {res[1]:8.3f} {res[2]:8.3f}')
    for res in residuals:
        for dim in range(ndim):
            if dim == ndim - 1:
                endline = "\n"
            else:
                endline = " "
            print(f"{res[dim]:8.3f}", end=endline)

    print("-------------------------------")
    print(f"Covariance matrix:")
    for var in result.var_names:
        if var is result.var_names[-1]:
            endline = "\n"
        else:
            endline = " "
        print(f"   {var:7s}", end=endline)

    for row in result.covar:
        for cov in row:
            if cov == row[-1]:
                endline = "\n"
            else:
                endline = " "
            print(f"{cov:10.5f}", end=endline)
