from ast import Assert
import numpy as np
import cv2
from typing import List, Tuple

from lib.classes import Camera
from lib.sfm.triangulation import Triangulate
from lib.misc import convert_to_homogeneous, convert_from_homogeneous


from thirdparty.transformations import affine_matrix_from_points


''' Space resection class for orienting one single image in world space'''


class Space_resection():
    def __init__(self, camera: Camera,
                 ) -> None:
        self.camera = camera

    def estimate(self,
                 image_points: np.ndarray,
                 object_poits: np.ndarray,
                 reprojection_error:  float = 3.0,
                 ) -> None:
        ret, r, t, inliers = cv2.solvePnPRansac(object_poits,
                                                image_points,
                                                self.camera.K,
                                                self.camera.dist,
                                                reprojectionError=reprojection_error
                                                )
        if ret:
            print(
                f'Space resection succeded. Number of inlier points: {len(inliers)}/{len(object_poits)}')
        else:
            print('Space resection failed. Wrong input data or not enough inliers found')
            return

        R, _ = cv2.Rodrigues(r)
        extrinsics = np.concatenate((R, t), axis=1)
        self.camera.build_camera_EO(extrinsics=extrinsics)


''' Absolute orientation class for georeferencing model'''


class Absolute_orientation():
    # @TODO: Apply transformation also to cameras!
    def __init__(self,
                 cameras: Tuple[Camera],
                 points3d_final: np.ndarray = None,
                 points3d_orig: np.ndarray = None,
                 image_points: Tuple[np.ndarray] = None,
                 camera_centers_world: Tuple[np.ndarray] = None,
                 ) -> None:

        self.cameras = cameras
        if points3d_final is not None:
            self.v1 = points3d_final
        else:
            raise ValueError(
                'Missing input for points in world reference system. Please, provide the their 3D coordinates in nx3 numpy array.')
        if points3d_orig is not None:
            self.v0 = points3d_orig
        elif image_points is not None:
            self.v0 = self.triangulate_image_points(image_points)
        else:
            raise ValueError(
                'Missing input for points in local reference system. Please, provide the their 3D coordinates or the image points to be triangulated')
        if camera_centers_world is not None:
            self.add_camera_centers_to_points(camera_centers_world)

    def add_camera_centers_to_points(
        self,
        camera_centers_world: List,
        v0: np.ndarray = None,
        v1: np.ndarray = None,
    ) -> None:
        ''' Add camera centers to arrays of points for estimating transformation.
        --------------
        Parameters:
            - camera_centers_world (List[np.ndarray]): List of numpy array containing the two camera centers.
            - v0 (nx3 np.ndarray, default=None): points in the original RS. If provided, they overwrite the old point stored in the class object.
            - v1 (nx3 np.ndarray, default=None): points in the final RS. If provided, they overwrite the old point stored in the class object.
        Returns: None
        --------------
        '''

        # if new arrays of points are provided, overwrite old ones.
        if v0:
            self.v1 = v0
        if v1:
            self.v0 = v1

        if camera_centers_world is None:
            raise ValueError(
                'Missing camera_centers_world argument. Please, provide Tuple with coordinates of the camera centers in world reference system to be added')
        self.v0 = np.concatenate(
            (self.v0,
                #  @TODO: add possibility of using multiple cameras
                self.cameras[0].C.reshape(1, -1),
                self.cameras[1].C.reshape(1, -1),
             )
        )
        # print(f'v0: {self.v0}')
        self.v1 = np.concatenate(
            (self.v1, camera_centers_world)
        )
        # print(f'v1: {self.v1}')

    def triangulate_image_points(self,
                                 image_points: List[np.ndarray]
                                 ) -> np.ndarray:
        #  @TODO: add possibility of using multiple cameras

        triangulation = Triangulate(
            self.cameras,
            image_points,
        )
        triangulation.triangulate_two_views()
        return triangulation.points3d

    def estimate_transformation_linear(self,
                                       estimate_scale:  bool = False,
                                       ) -> np.ndarray:
        ''' Wrapper around 'affine_matrix_from_points' function from 'transformation'. It estimates 3D rototranslation by using SVD
        '''

        self.tform = affine_matrix_from_points(
            self.v0.T,
            self.v1.T,
            shear=False,
            scale=estimate_scale,
            usesvd=False
        )
        # print(f'Estimated transformation: \n{self.tform}')

        return self.tform

    def estimate_transformation_least_squares(self):
        pass

    def apply_transformation(self,
                             T: np.ndarray = None,
                             points3d: np.ndarray = None,
                             camera: Camera = None,
                             ) -> np.ndarray:
        ''' Apply estimated transformation to 3D points and to camera matrices
        --------------
        Parameters:
            - T (4x4 np.ndarray = None): 4x4 transformation matrix. If provided, they overwrite the self.tform.
            - points3d (nx3 np.ndarray = None): Coordinates of the points to be transformed. If provided, they overwrite the self.v1 points.
            - camera (Camera, default=None): Camera object. If provided, it overwrites the old point stored in the class object.
        Returns: self.v1 (nx3 np.ndarray): transformed points.
        --------------
        '''
        assert not(self.v1 is None and points3d is None), f'Points to be transformed not found in self.v1 and not provided. Please provide a set of points to be transformed.'

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
                camera.pose = T @ camera.pose
                camera.pose_to_extrinsics()
                camera.update_camera_from_extrinsics()
        else:
            camera.pose = T @ camera.pose
            camera.pose_to_extrinsics()
            camera.update_camera_from_extrinsics()

        # print('Cam 1:\n', self.cameras[0].extrinsics)
        # print('Cam 2:\n', self.cameras[1].extrinsics)

        return self.v1
