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
                 points3d_world: np.ndarray = None,
                 points3d_loc: np.ndarray = None,
                 image_points: Tuple[np.ndarray] = None,
                 ) -> None:

        self.cameras = cameras
        if points3d_world is not None:
            self.points3d_world = points3d_world
        else:
            raise ValueError(
                'Missing input for points in world reference system. Please, provide the their 3D coordinates in nx3 numpy array.')
        if points3d_loc is not None:
            self.points3d_loc = points3d_loc
        elif image_points is not None:
            self.points3d_loc = self.triangulate_image_points(image_points)
        else:
            raise ValueError(
                'Missing input for points in local reference system. Please, provide the their 3D coordinates or the image points to be triangulated')

    def triangulate_image_points(self,
                                 image_points: List[np.ndarray]
                                 ):  # -> np.ndarray:
        #  @TODO: add possibility of using multiple cameras

        triangulation = Triangulate(
            self.cameras,
            image_points,
        )
        triangulation.triangulate_two_views()
        return triangulation.points3d

    def estimate_transformation(self,
                                v0: np.ndarray = None,
                                v1: np.ndarray = None,
                                estimate_scale:  bool = False,
                                add_camera_centers: bool = False,
                                camera_centers_world: Tuple = None,
                                ) -> np.ndarray:
        ''' Wrapper around 'affine_matrix_from_points' function from 'transformation' 
        '''
        if v0 is None:
            v0 = self.points3d_loc
        if v1 is None:
            v1 = self.points3d_world

        if add_camera_centers:
            if camera_centers_world is None:
                raise ValueError(
                    'Missing camera_centers_world argument. Please, provide Tuple with coordinates of the camera centers in world reference system to be added')
            v0 = np.concatenate(
                (v0,
                 #  @TODO: add possibility of using multiple cameras
                 self.cameras[0].C.reshape(1, -1),
                 self.cameras[1].C.reshape(1, -1),
                 )
            )
            # print(f'v0: {v0}')
            v1 = np.concatenate(
                (v1, camera_centers_world)
            )
            # print(f'v1: {v1}')
            self.points3d_loc = v0.copy()
            self.points3d_world = v1.copy()

        self.tform = affine_matrix_from_points(
            v0.T,
            v1.T,
            shear=False,
            scale=estimate_scale,
            usesvd=True
        )
        # print(f'Estimated transformation: \n{self.tform}')
        return self.tform

    def apply_transformation(self,
                             points3d: np.ndarray,
                             ) -> np.ndarray:
        # @TODO: Apply transformation also to cameras!

        points3d = convert_to_homogeneous(points3d.T)
        points_out = self.tform @ points3d
        return convert_from_homogeneous(points_out).T
