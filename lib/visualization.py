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
import open3d as o3d

from lib.classes import (Camera, Features)
from lib.geometry import (undistort_points,
                          project_points,
                          )

''' Misc functions'''


def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))


''' Visualization of features and matches on images'''


def make_matching_plot(image0, image1, pts0, pts1,
                       pts_col=(0, 0, 255), point_size=2, line_col=(0, 255, 0),
                       line_thickness=1, path=None, margin=10):
    if image0.ndim > 2:
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    if image1.ndim > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    mkpts0, mkpts1 = np.round(pts0).astype(int), np.round(pts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=line_col, thickness=line_thickness, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), point_size,
                   pts_col, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), point_size, pts_col, -1,
                   lineType=cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(path, out)


def plot_features(image, features, title: str = None, ax=None):
    ''' Plot detected features on the input image
    Parameters
    ----------
    image : numpy array with BRG channels (OpenCV standard)
    features : nx2 float32 array
        array of 2D image coordinates of the features to plot
    title: str
        title of the axes of the plt
    ax : matplotlib axes (default = None)
        axis in which to make the plot. If nothing is given, the function create
        a new figure and axes.
    Return : None
    '''
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(im)
    ax.scatter(features[:, 0], features[:, 1],
               s=6, c='y', marker='o',
               alpha=0.8, edgecolors='r',
               linewidths=1,
               )
    if title is not None:
        ax.set_title(title)


def plot_projections(points3d, camera: Camera, image, title: str = None, ax=None):
    '''  Project 3D point to a camera and plot projections over the image
    Parameters
    ----------
    points3d: nx3 float32 array
        Array of 3D points in the object space
    camera: Camera object
        Camera object contining exterior and interior orientation
    image: numpy array with BRG channels (OpenCV standard)
    title: str
        title of the axes of the plt
    ax : matplotlib axes (default = None)
        axis in which to make the plot. If nothing is given, the function create
        a new figure and axes.
    Return : None
    '''
    if ax is None:
        fig, ax = plt.subplots()

    projections = project_points(points3d, camera)
    plot_features(image, projections, title, ax)


def draw_epip_lines(img0, img1, lines, pts0, pts1, fast_viz=True):
    ''' img0 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img0.shape
    if not fast_viz:
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # TODO: implement visualization in matplotlib
    for r, pt0, pt1 in zip(lines, pts0, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img0 = cv2.line(img0, (x0, y0), (x1, y1), color, 1)
        img0 = cv2.circle(img0, tuple(pt0.astype(int)), 5, color, -1)
        img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
        # img0 = cv2.drawMarker(img0,tuple(pt0.astype(int)),color,cv2.MARKER_CROSS,3)
        # img1 = cv2.drawMarker(img1,tuple(pt1.astype(int)),color,cv2.MARKER_CROSS,3)
    return img0, img1


''' Visualization of 3D point clouds'''


def display_point_cloud(pcd, cameras: list = None,
                        viz_rs: bool = True,
                        win_name: str = 'Point cloud',
                        ) -> None:
    ''' Display a O3D point cloud
    Parameters
    ----------
    pcd : O3D obejct
        Point cloud with n points.
    cameras : List of Camera objects (default = None)
        List of Camera objects, used to visualize the location 
        and orientation of the cameras in the plot. 
        If None is given, only the point cloud is plotted.
    Returns
    -------
    None.
    '''

    if cameras is not None:
        num_cams = len(cameras)
        if num_cams < 3:
            cam_colors = [[1, 0, 0], [0, 0, 1]]
        else:
            cam_colors = np.full((num_cams, 3), [1, 0, 0])

        cam_syms = []
        for i, cam in enumerate(cameras):
            cam_syms.append(make_camera_pyramid(cam,
                                                color=cam_colors[i],
                                                focal_len_scaled=30,
                                                ))

    o3d.visualization.draw_geometries([pcd,  cam_syms[0], cam_syms[1]], window_name=win_name,
                                      width=1280, height=720,
                                      left=300, top=200)


def display_pc_inliers(cloud, ind):
    ''' Display a O3D point cloud, separating inliers from outliers 
    (e.g. after a SOR filter)
    Parameters
    ----------
    cloud : O3D obejct
        Point cloud with n points.
    ind : (nx1) List of int
        List of indices of the inlier points
    Returns
    -------
    None.
    '''
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def make_camera_pyramid(camera: Camera,
                        color=[1., 0., 0.],
                        focal_len_scaled=5,
                        aspect_ratio=0.3
                        ):
    '''
    Parameters
    ----------
    camera : Camera object 
    color : list of float32
        Color of the pyramid. The default is [1., 0., 0.],
    focal_len_scaled : float32, optional
        Scale for visualizing the camera pyramid. The default is 5.
    aspect_ratio : float32, optional
        Aspect ration of the pyramid for visualization. The default is 0.3.

    Returns
    -------
    o3d_line : o3d geometry
        Open3D geometry object containing the camera pyramid to be plotted.

    '''
    # Check if camera pose is available, otherwise build it.
    if camera.pose is None:
        camera.Rt_to_extrinsics()
        camera.extrinsics_to_pose()
    vertexes = pose2pyramid(camera.pose, focal_len_scaled, aspect_ratio)

    # Build Open3D camera object
    vertex_pcd = o3d.geometry.PointCloud()
    vertex_pcd.points = o3d.utility.Vector3dVector()

    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]
    color = np.array(color, dtype='float32')
    colors = [color for i in range(len(lines))]

    cam_view_obj = o3d.geometry.LineSet()
    cam_view_obj.lines = o3d.utility.Vector2iVector(lines)
    cam_view_obj.colors = o3d.utility.Vector3dVector(colors)
    cam_view_obj.points = o3d.utility.Vector3dVector(vertexes)

    return cam_view_obj


def pose2pyramid(camera_pose, focal_len_scaled=5, aspect_ratio=0.3):
    # TODO: add description
    '''Function inspired from https://github.com/demul/extrinsic2pyramid

    Parameters
    ----------
    camera_pose : float32 array
        Camera pose matrix.
    focal_len_scaled : float32, optional
        Scale for visualizing the camera pyramid. The default is 5.
    aspect_ratio : float32, optional
        Aspect ration of the pyramid for visualization. The default is 0.3.

    Returns
    -------
    vertex_transformed: matrix of the vertexes of the pyramid in the world reference system

    '''
    vertex_std = np.array([[0, 0, 0, 1],
                           [focal_len_scaled * aspect_ratio, -focal_len_scaled *
                               aspect_ratio, focal_len_scaled, 1],
                           [focal_len_scaled * aspect_ratio, focal_len_scaled *
                               aspect_ratio, focal_len_scaled, 1],
                           [-focal_len_scaled * aspect_ratio, focal_len_scaled *
                               aspect_ratio, focal_len_scaled, 1],
                           [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]],
                          dtype=np.float32)
    vertex_transformed = vertex_std @ camera_pose.T

    return vertex_transformed[:, 0:3]


def make_viz_sdr(scale=5):
    '''
    Parameters
    ----------
    scale : int
        scale for plotting the axes vectors
    Returns
    -------
    rs_obj : o3d geometry
        Open3D geometry object containing the reference system visualization symbol
        Colors are R, G, B respectively for X, Y, Z axis
    '''
    vertexes = scale * np.array([[0., 0., 0.],
                                 [1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.],
                                 ])
    colors = np.array([[255., 0., 0.],
                      [0., 255., 0.],
                      [0., 0., 255.]],
                      dtype='float32')
    lines = [[0, 1], [0, 2], [0, 3]]

    rs_obj = o3d.geometry.LineSet()
    rs_obj.lines = o3d.utility.Vector2iVector(lines)
    rs_obj.colors = o3d.utility.Vector3dVector(colors)
    rs_obj.points = o3d.utility.Vector3dVector(vertexes)

    return rs_obj


''' Other skatched functions to be implemented'''
# Darw features
# img0 = images[1][0]
# pt0 = features[1][epoch].get_keypoints()
# # pts_col = tuple(np.random.randint(0,255,3).tolist())
# pts_col=(0,255,0)
# point_size=2
# # img0_features = cv2.drawMarker(img0,tuple(pt0.astype(int)),color,cv2.MARKER_CROSS,3)
# for (x0, y0) in pt0.astype(int):


# # Plot with OpenCV
# cam = 0
# i = 0
# xy = targets.get_im_coord(cam)[i]
# img = images[cam][i]
# cv2.namedWindow(f'cam {cam}, epoch {i}', cv2.WINDOW_NORMAL)
# color=(0,255,0)
# point_size=2
# img_target = cv2.drawMarker(img,tuple(xy.astype(int)),color,cv2.MARKER_CROSS,1)
# cv2.imshow(f'cam {cam}, epoch {i}', img_target)
# cv2.waitKey()
# cv2.destroyAllWindows()

# #  Plot with matplotlib
# cam = 1
# i = 0
# _plot_style = dict(markersize=5, markeredgewidth=2,
#                    markerfacecolor='none', markeredgecolor='r',
#                    marker='x', linestyle='none')
# xy = targets.get_im_coord(cam)[i]
# img = cv2.cvtColor(images[cam][i], cv2.COLOR_BGR2RGB)
# fig, ax = plt.subplots()
# ax.imshow(img)
# ax.plot(xy[0], xy[1], **_plot_style)


# x = np.arange(-10,10)
# y = x**2

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x,y)

# coords = []

# def onclick(event):
#     global ix, iy
#     ix, iy = event.xdata, event.ydata
#     print(f'x = {ix}, y = {iy}')

#     global coords
#     coords.append((ix, iy))

#     if len(coords) == 2:
#         fig.canvas.mpl_disconnect(cid)

#     return coords
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
