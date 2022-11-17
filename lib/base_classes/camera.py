import numpy as np

from typing import List, Union, Tuple
from scipy import linalg
from pathlib import Path

from lib.import_export.importing import read_opencv_calibration


class Camera:
    """Class to manage Pinhole Cameras."""

    def __init__(
        self,
        width: np.ndarray,
        height: np.ndarray,
        K: np.ndarray = None,
        dist: np.ndarray = None,
        R: np.ndarray = None,
        t: np.ndarray = None,
        extrinsics: np.ndarray = None,
        calib_path: Union[str, Path] = None,
    ):
        """Initialize pinhole camera model
        All the Camera members are private in order to guarantee consistency
        between the different possible expression of the exterior orientation.
        Use ONLY Getter and Setter method to access to Camera parameters from
        outside the Class.

        For code safety, use the method "update_extrinsics" to update Camera
        Exterior Orientataion given a new extrinsics matrix. If you need to
        update the camera EO from a pose matrix or from R,t, compute the
        extrinsics matrix first with the methods Camera.pose_to_extrinsics
        (pose) or Camera.Rt_to_extrinsics(R,t), that return the extrinsics
        matrix.
        """

        self._w = width  # Image width [px]
        self._h = height  # Image height [px]g
        self._K = K  # Calibration matrix (Intrisics)
        self._dist = dist  # Distortion vector in OpenCV format
        self.reset_EO()

        if R is not None and t is not None:
            self._extrinsics = self.Rt_to_extrinsics(R, t)

        if extrinsics is not None:
            self._extrinsics = extrinsics

        # If calib_path is provided, read camera calibration from file
        if calib_path is not None:
            self.read_calibration_from_file(calib_path)

    # Getters
    @property
    def width(self) -> np.ndarray:
        """Get image width"""
        return self._w

    @property
    def height(self) -> np.ndarray:
        """Get image height"""
        return self._h

    @property
    def K(self) -> np.ndarray:
        """Get Intrinsics matrix"""
        return self._K

    @property
    def dist(self) -> np.ndarray:
        """ " Get non-linear distortion parameter vector"""
        return self._dist

    @property
    def extrinsics(self) -> np.ndarray:
        """Get Extrinsics Matrix (i.e., transformation from world to camera)"""
        return self._extrinsics

    @property
    def pose(self) -> np.ndarray:
        """Get Pose Matrix (i.e., transformation from camera to world)"""
        return self.extrinsics_to_pose()

    @property
    def C(self) -> np.ndarray:
        """Get camera center (i.e., coordinates of the projective centre in world reference system, that is the translation from camera to world)"""
        pose = self.extrinsics_to_pose()
        return pose[0:3, 3:4]

    @property
    def t(self) -> np.ndarray:
        """Get translation vector (i.e., translation from world to camera)"""
        return self._extrinsics[0:3, 3:4]

    @property
    def R(self) -> np.ndarray:
        """Get rotation matrix (i.e., rotation matrix from world to camera)"""
        return self._extrinsics[0:3, 0:3]

    @property
    def euler_angles(self) -> Tuple[float]:
        """Get Omega Phi and Kappa rotation angles (rotations from world to camera)"""
        return self.euler_from_R(self.R)

    @property
    def P(self) -> np.ndarray:
        """Get Projective matrix P = K [ R | T ]"""
        return self.K @ self._extrinsics[0:3, :]

    # Setters
    def update_K(self, K: np.ndarray) -> None:
        self._K = K

    def update_dist(self, dist: np.ndarray) -> None:
        self._dist = dist

    def update_extrinsics(self, extrinsics: np.ndarray) -> None:
        """Update Camera Exterior Orientataion given a new extrinsics matrix.
        Note that, for code safety, this is the only method to update the camera Exterior Orientation.
        If you need to update the camera EO from a pose matrix or from R,t, compute the extrinsics matrix first with the method self.pose_to_extrinsics(pose) or Rt_to_extrinsics(R,t)
        """
        self._extrinsics = extrinsics

    # Methods
    def reset_EO(self) -> None:
        """Reset camera External Orientation (EO), in such a way as to make camera reference system parallel to world reference system"""
        self._extrinsics = np.eye(4)

    def read_calibration_from_file(self, path: Union[str, Path]) -> None:
        """
        Read camera internal orientation from file and save in camera class.
        The file must contain the full K matrix and distortion vector,
        according to OpenCV standards, and organized in one line, as follow:
        width height fx 0. cx 0. fy cy 0. 0. 1. k1, k2, p1, p2, [k3, [k4, k5, k6
        Values must be float(include the . after integers) and divided by a
        white space.
        -------
        Returns: None
        """

        w, h, K, dist = read_opencv_calibration(path)
        self._width = w
        self._height = h
        self._K = K
        self._dist = dist

    def extrinsics_to_pose(self, extrinsics: np.ndarray = None) -> np.ndarray:
        """Compute Pose matrix (i.e., transformation from camera to world) from extrinsics matrix (i.e, transformation from world to camera)"""

        if extrinsics is None:
            extrinsics = self._extrinsics

        R = extrinsics[0:3, 0:3]
        t = extrinsics[0:3, 3:4]
        Rc = R.T
        C = -np.dot(Rc, t)
        Rc_block = self.build_block_matrix(Rc)
        C_block = self.build_block_matrix(C)

        return np.dot(C_block, Rc_block)

    def pose_to_extrinsics(self, pose: np.ndarray):
        """Returns Pose matrix given an extrinsics matrix"""

        Rc = pose[0:3, 0:3]
        C = pose[0:3, 3:4]
        R = Rc.T
        t = -R @ C
        t_block = self.build_block_matrix(t)
        R_block = self.build_block_matrix(R)

        return t_block @ R_block

    def factor_P(self) -> Tuple[np.ndarray]:
        """Factorize the camera matrix into K, R, t as P = K[R | t]."""

        # factor first 3*3 part
        K, R = linalg.rq(self.P[:, :3])

        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1, 1] *= -1

        K = np.dot(K, T)
        R = np.dot(T, R)  # T is its own inverse
        t = np.dot(linalg.inv(K), self.P[:, 3]).reshape(3, 1)

        return K, R, t

    def Rt_to_extrinsics(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Return 4x4 Extrinsics matrix, given a 3x3 Rotation matrix and 3x1 translation vector, as follows:

        [ R | t ]    [ I | t ]   [ R | 0 ]
        | --|-- |  = | --|-- | * | --|-- |
        [ 0 | 1 ]    [ 0 | 1 ]   [ 0 | 1 ]
        """
        if len(t.shape) == 1 or t.shape == (1, 3):
            assert t.shape[0] == 3, "Invalid translation vector"
            t = t.reshape(3, 1)
        R_block = self.build_block_matrix(R)
        t_block = self.build_block_matrix(t)
        return t_block @ R_block

    def C_from_P(self, P: np.ndarray) -> np.ndarray:
        """
        turn the camera center from a projection matrix P, as
        C = [- inv(KR) * Kt] = [-inv(P[1:3]) * P[4]]
        """
        return -np.dot(np.linalg.inv(P[:, 0:3]), P[:, 3].reshape(3, 1))

    def build_pose_matrix(self, R: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Return the Pose Matrix given R and C"""
        # Check for input dimensions
        if R.shape != (3, 3):
            raise ValueError(
                "Wrong dimension of the R matrix. It must be a 3x3 numpy array"
            )
        if C.shape == (3,) or C.shape == (1, 3):
            C = C.T
        elif C.shape != (3, 1):
            raise ValueError(
                "Wrong dimension of the C vector. It must be a 3x1 or a 1x3 numpy array"
            )

        pose = np.eye(4)
        pose[0:3, 0:3] = R
        pose[0:3, 3:4] = C
        return pose

    def euler_from_R(self, R):
        """
        Compute Euler angles from rotation matrix
        - ------
        Returns:  [omega, phi, kappa]
        """
        omega = np.arctan2(R[2, 1], R[2, 2])
        phi = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        kappa = np.arctan2(R[1, 0], R[0, 0])

        return (omega, phi, kappa)

    def build_block_matrix(self, mat):
        if mat.shape[1] == 3:
            block = np.block([[mat, np.zeros((3, 1))], [np.zeros((1, 3)), 1]])
        elif mat.shape[1] == 1:
            block = np.block([[np.eye(3), mat], [np.zeros((1, 3)), 1]])
        else:
            print("Error: unknown input matrix dimensions.")
            return None

        return block

    def make_mat_homogeneous(self, mat) -> np.ndarray:
        """
        Return a homogeneous matrix, given a euclidean matrix (i.e., it adds a row of zeros with the last elemmatrixent as 1)
            [     mat    ]
            [------------]
            [ 0  0  0  1 ]
        """
        n, m = mat.shape
        mat_h = np.eye(n + 1, m)
        mat_h[0:n, 0:m] = mat

        return mat_h
