import numpy as np
import cv2

from typing import List, Tuple
from lmfit import Minimizer, Parameters, fit_report
from scipy import stats

from .triangulation import Triangulate
from ..classes.camera import Camera
from ..utils.utils import convert_to_homogeneous, convert_from_homogeneous

from ..thirdparty.transformations import (
    affine_matrix_from_points,
    euler_from_matrix,
    euler_matrix,
)


""" Space resection class for orienting one single image in world space"""


class Space_resection:
    def __init__(
        self,
        camera: Camera,
    ) -> None:
        self.camera = camera

    def estimate(
        self,
        image_points: np.ndarray,
        object_poits: np.ndarray,
        reprojection_error: float = 3.0,
    ) -> None:
        ret, r, t, inliers = cv2.solvePnPRansac(
            object_poits,
            image_points,
            self.camera.K,
            self.camera.dist,
            reprojectionError=reprojection_error,
        )
        if ret:
            print(
                f"Space resection succeded. Number of inlier points: {len(inliers)}/{len(object_poits)}"
            )
        else:
            print(
                "Space resection failed. Wrong input data or not enough inliers found"
            )
            return

        R, _ = cv2.Rodrigues(r)
        extrinsics = np.concatenate((R, t), axis=1)
        self.camera.build_camera_EO(extrinsics=extrinsics)


""" Absolute orientation class for georeferencing model"""


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


class Absolute_orientation:
    def __init__(
        self,
        cameras: Tuple[Camera],
        points3d_final: np.ndarray,
        points3d_orig: np.ndarray = None,
        image_points: Tuple[np.ndarray] = None,
        camera_centers_world: Tuple[np.ndarray] = None,
    ) -> None:
        """
        __init__ Initialize Absolute Orientation class

        Args:
            cameras (Tuple[Camera]): Tuple containing Camera objects
            points3d_final (np.ndarray, optional): nx3 array containing object coordinates in final reference system.
            points3d_orig (np.ndarray, optional): nx3 array containing object coordinates in original reference system (to be transformed by the estimated Helmert transformation). It is possible to omit this, if image coordinates of the corresponding points to be triangulated are provided. Defaults to None.
            image_points (Tuple[np.ndarray], optional): Tuple of numpy nx2 array containing image points to be trinagulated. If provided, a linear triangulation is computed and the resulting 3D points will be used for points3d_orig. Defaults to None.
            camera_centers_world (Tuple[np.ndarray], optional): Tuple of nx3 array containing camera centers in final reference system. If provided, they are concatenate to points3d_final. Defaults to None.

        """
        self.cameras = cameras
        if points3d_final is not None and points3d_final.shape[1] == 3:
            self.v1 = points3d_final
        else:
            raise ValueError(
                "Missing or wrong input for points in world reference system. Please, provide the their 3D coordinates in nx3 numpy array."
            )
        if points3d_orig is not None:
            self.v0 = points3d_orig
        elif image_points is not None:
            self.v0 = self.triangulate_image_points(image_points)
        else:
            raise ValueError(
                "Missing input for points in local reference system. Please, provide the their 3D coordinates or the image points to be triangulated"
            )
        if camera_centers_world is not None:
            self.add_camera_centers_to_points(camera_centers_world)

    def add_camera_centers_to_points(
        self,
        camera_centers_world: List,
        v0: np.ndarray = None,
        v1: np.ndarray = None,
    ) -> None:
        """Add camera centers to arrays of points for estimating transformation.
        --------------
        Parameters:
            - camera_centers_world (List[np.ndarray]): List of numpy array containing the two camera centers.
            - v0 (nx3 np.ndarray, default=None): points in the original RS. If provided, they overwrite the old point stored in the class object.
            - v1 (nx3 np.ndarray, default=None): points in the final RS. If provided, they overwrite the old point stored in the class object.
        Returns: None
        --------------
        """

        # if new arrays of points are provided, overwrite old ones.
        if v0:
            self.v1 = v0
        if v1:
            self.v0 = v1

        if camera_centers_world is None:
            raise ValueError(
                "Missing camera_centers_world argument. Please, provide Tuple with coordinates of the camera centers in world reference system to be added"
            )
        self.v0 = np.concatenate(
            (
                self.v0,
                #  @TODO: add possibility of using multiple cameras
                self.cameras[0].C.reshape(1, -1),
                self.cameras[1].C.reshape(1, -1),
            )
        )
        # print(f'v0: {self.v0}')
        self.v1 = np.concatenate((self.v1, camera_centers_world))
        # print(f'v1: {self.v1}')

    def triangulate_image_points(self, image_points: List[np.ndarray]) -> np.ndarray:
        #  @TODO: add possibility of using multiple cameras
        triangulation = Triangulate(
            self.cameras,
            image_points,
        )
        triangulation.triangulate_two_views()
        return triangulation.points3d

    def estimate_transformation_linear(
        self,
        estimate_scale: bool = True,
    ) -> np.ndarray:
        """Wrapper around 'affine_matrix_from_points' function from 'transformation'. It estimates 3D rototranslation by using SVD"""

        self.tform = affine_matrix_from_points(
            self.v0.T, self.v1.T, shear=False, scale=estimate_scale, usesvd=False
        )
        # print(f'Estimated transformation: \n{self.tform}')

        return self.tform

    def extract_params_from_T(self, T: np.ndarray = None):
        """Extract transformation parameters from 4x4 T matrix
        --------------
        Parameters:
            - T (4x4 np.ndarray = None): 4x4 transformation matrix. If not provided, self.tform is used.
        Returns: params (dict): transformation parameters.
        --------------
        """

        if T is None:
            T = self.tform

        t = T[:3, 3:4].squeeze()
        rot = euler_from_matrix(T[:3, :3])
        m = float(1.0)
        prm = {
            "rx": rot[0],
            "ry": rot[1],
            "rz": rot[2],
            "tx": t[0],
            "ty": t[1],
            "tz": t[2],
            "m": m,
        }

        return prm

    def estimate_transformation_least_squares(
        self,
        uncertainty: np.ndarray = None,
    ) -> np.ndarray:
        """Estimate 3D rototranslation with least squares, by using lmfit library.
        --------------
        Parameters:
            - uncertainty (np.array): numpy array of the same dimension of self.v0 and self.v1 matrixes used to scale the residuals. The uncertainty matrix contains the a priori standard deviation of each observation and it is analogous as building a diagonal Q matrix with the variance of the observations along the main diagonal.
        Returns: self.tform (4x4 np.ndarray): Transformation matrix.
        --------------
        """
        # @TODO: storing and returning also covariance matrix and other useful information.

        # Estimate approximate values by using estimate_transformation_linear
        T = self.estimate_transformation_linear()
        prm = self.extract_params_from_T(T)

        # Define Parameters to be optimized
        params = Parameters()
        params.add("rx", value=prm["rx"], vary=True)
        params.add("ry", value=prm["ry"], vary=True)
        params.add("rz", value=prm["rz"], vary=True)
        params.add("tx", value=prm["tx"], vary=True)
        params.add("ty", value=prm["ty"], vary=True)
        params.add("tz", value=prm["tz"], vary=True)
        params.add("m", value=prm["m"], vary=True)

        if uncertainty is None:
            # Default assigned uncertainty
            uncertainty = np.ones(self.v0.shape)

        # Run Optimization!
        weights = 1.0 / uncertainty
        minimizer = Minimizer(
            compute_residuals,
            params,
            fcn_args=(
                self.v0,
                self.v1,
            ),
            fcn_kws={
                "weights": weights,
            },
            scale_covar=True,
        )
        ls_result = minimizer.minimize(method="leastsq")
        # fit_report(result)

        # Print result
        print_results(ls_result, weights)

    def apply_transformation(
        self,
        T: np.ndarray = None,
        points3d: np.ndarray = None,
        camera: Camera = None,
    ) -> np.ndarray:
        """Apply estimated transformation to 3D points and to camera matrices
        --------------
        Parameters:
            - T (4x4 np.ndarray = None): 4x4 transformation matrix. If not provided, self.tform is used.
            - points3d (nx3 np.ndarray = None): Coordinates of the points to be transformed.If not provided, self.v1 is used.
            - camera (Camera, default=None): Camera object. If not provided, self.cameras are used.
        Returns: self.v1 (nx3 np.ndarray): transformed points.
        --------------
        """
        assert not (
            self.v1 is None and points3d is None
        ), f"Points to be transformed not found in self.v1 and not provided. Please provide a set of points to be transformed."

        if T is None:
            T = self.tform

        if points3d is None:
            points3d = self.v1

        # Apply transformation to points
        points_out = T @ convert_to_homogeneous(points3d.T)
        self.v1 = convert_from_homogeneous(points_out).T

        # Apply transformation to cameras
        if camera is None:
            for camera in self.cameras:
                pose = T @ camera.pose
                extrinsics = camera.pose_to_extrinsics(pose)
                camera.update_extrinsics(extrinsics)
        else:
            pose = T @ camera.pose
            extrinsics = camera.pose_to_extrinsics(pose)
            camera.update_extrinsics(extrinsics)

        # print('Cam 1:\n', self.cameras[0].extrinsics)
        # print('Cam 2:\n', self.cameras[1].extrinsics)

        return self.v1
