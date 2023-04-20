import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import matplotlib.colors as Colors
import matplotlib.cm as cm
import matplotlib
import logging

from typing import List, Union, Dict
from pathlib import Path
from copy import deepcopy

from icepy4d.classes.camera import Camera
from icepy4d.classes.features import Features, Feature
from icepy4d.classes.point_cloud import PointCloud
from icepy4d.sfm.geometry import project_points


""" Visualization of images"""


def imshow_cv2(
    img: np.ndarray, win_name: str = None, convert_RGB2BRG: bool = True
) -> None:
    """Wrapper for visualizing an image with OpenCV

    Args:
        img (np.ndarray): Input image.
        win_name (str): Name of the window. Default is None.
        convert_RGB2BRG (bool): Whether to convert RGB to BGR. Default is True.

    Returns:
        None: This function does not return anything.
    """
    if win_name is None:
        win_name = "image"
    if convert_RGB2BRG:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def plot_image_pair(
    imgs: List[np.ndarray], dpi: int = 100, size: float = 6, pad: float = 0.5
) -> plt.figure:
    """Plot a pair of images.

    Args:
        imgs (List[np.ndarray]): A list containing two images.
        dpi (int): Dots per inch. Default is 100.
        size (float): Figure size. Default is 6.
        pad (float): Padding between subplots. Default is 0.5.

    Returns:
        plt.figure: Figure object of the plotted images.
    """

    n = len(imgs)
    assert n == 2, "number of images must be two"
    figsize = (size * n, size * 3 / 4) if size is not None else None
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap("gray"), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)
    return fig


""" Visualization of features and matches on images"""


def plot_keypoints(kpts0, kpts1, color="w", ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def draw_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    color = np.array(color)
    if color.shape == (3,):
        color = np.repeat(color.reshape(1, 3), len(kpts0), axis=0)
    if color.shape != (len(kpts0), 3):
        raise ValueError("invalid color input.")
    if color.dtype == np.int64:
        color = color / 255.0

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [
        matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]),
            (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1,
            transform=fig.transFigure,
            c=color[i],
            linewidth=lw,
        )
        for i in range(len(kpts0))
    ]


def plot_matches(
    image0,
    image1,
    pts0,
    pts1,
    color: List[int] = [0, 255, 0],
    point_size=1,
    line_thickness=1,
    path=None,
    fast_viz: bool = False,
) -> None:
    """Plot matching points between two images.

    Args:
        image0: The first image.
        image1: The second image.
        pts0: List of 2D points in the first image.
        pts1: List of 2D points in the second image.
        color: RGB color of the matching lines.
        point_size: Size of the circles representing the points.
        line_thickness: Thickness of the lines connecting the points.
        path: Path to save the output image.
        fast_viz: If True, use OpenCV to display the image.

    Returns:
        None.
    """

    if fast_viz:
        plot_matches_cv2(
            image0,
            image1,
            pts0,
            pts1,
            point_size=point_size,
            line_thickness=line_thickness,
            path=path,
        )
        return

    fig = plot_image_pair([image0, image1])
    plot_keypoints(pts0, pts1, color="r", ps=point_size)
    draw_matches(pts0, pts1, color, lw=line_thickness, ps=point_size)
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_matches_cv2(
    image0,
    image1,
    pts0,
    pts1,
    pts_col=(0, 0, 255),
    point_size=2,
    line_col=(0, 255, 0),
    line_thickness=1,
    path=None,
    margin=10,
) -> None:
    """Plot matching points between two images using OpenCV.

    Args:
        image0: The first image.
        image1: The second image.
        pts0: List of 2D points in the first image.
        pts1: List of 2D points in the second image.
        pts_col: RGB color of the points.
        point_size: Size of the circles representing the points.
        line_col: RGB color of the matching lines.
        line_thickness: Thickness of the lines connecting the points.
        path: Path to save the output image.
        margin: Margin between the two images in the output.

    Returns:
        None.
    """
    if image0.ndim > 2:
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    if image1.ndim > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin :] = image1
    out = np.stack([out] * 3, -1)

    mkpts0, mkpts1 = np.round(pts0).astype(int), np.round(pts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        cv2.line(
            out,
            (x0, y0),
            (x1 + margin + W0, y1),
            color=line_col,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
        # display line end-points as circles
        cv2.circle(out, (x0, y0), point_size, pts_col, -1, lineType=cv2.LINE_AA)
        cv2.circle(
            out, (x1 + margin + W0, y1), point_size, pts_col, -1, lineType=cv2.LINE_AA
        )
    if path is not None:
        cv2.imwrite(path, out)


def plot_points(
    image: np.ndarray,
    points: np.ndarray,
    title: str = None,
    ax: plt.Axes = None,
    save_path: Union[str, Path] = None,
    hide_fig: bool = False,
    zoom_to_features: bool = False,
    window_size: int = 50,
    **kwargs,
) -> plt.Axes:
    """
    plot_points  Plot points on input image.

    Args:
        image (np.ndarray): A numpy array with RGB channels.
        points (np.ndarray): An nx2 float32 array of 2D image coordinates of the features to plot.
        title (str, optional): The title of the plot. Defaults to None.
        ax (matplotlib.axes, optional): The axis in which to make the plot. If None, the function will create a new figure and axes. Defaults to None.
        save_path (Union[str, Path], optional): The path to save the plot. Defaults to None.
        hide_fig (bool, optional): Indicates whether to close the figure after plotting. Defaults to False.
        zoom_to_features (bool, optional): Indicates whether to zoom in to the features in the plot. Defaults to False.
        window_size (int, optional): The size of the zoom window. Defaults to 50.
        **kwargs: additional keyword arguments for plotting characteristics (e.g. `s`, `c`, `marker`, etc.). Refer to matplotlib.pyplot.scatter documentation for more information https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html.

    Returns:
        plt.Axes: matplotlib axis
    """
    s = 6
    c = "y"
    marker = "o"
    alpha = 0.8
    edgecolors = "r"
    linewidths = 1
    size_inches = (18.5, 10.5)  # (23.8, 15,75) #
    dpi = 600

    # overwrite default values with kwargs if provided
    s = kwargs.get("s", s)
    c = kwargs.get("c", c)
    marker = kwargs.get("marker", marker)
    alpha = kwargs.get("alpha", alpha)
    edgecolors = kwargs.get("edgecolors", edgecolors)
    linewidths = kwargs.get("linewidths", linewidths)
    size_inches = kwargs.get("size_inches", size_inches)
    dpi = kwargs.get("dpi", dpi)

    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=s,
        c=c,
        marker=marker,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths,
    )
    if title is not None:
        ax.set_title(title)

    if zoom_to_features:
        w = window_size  # px
        xc = points[:, 0].mean()
        yc = points[:, 1].mean()
        ax.set_xlim([xc - w, xc + w])
        ax.set_ylim([yc - w, yc + w])
    if save_path is not None:
        fig.set_size_inches(size_inches[0], size_inches[1])
        fig.savefig(save_path, dpi=dpi)
    if hide_fig is True:
        plt.close(fig)
        return None
    else:
        return ax


def plot_points_cv2(
    image: np.ndarray,
    points: np.ndarray,
    title: str = "figure",
    save_path: str = None,
) -> None:
    """
    Displays an OpenCV image with keypoints overlaid.

    Args:
        image (np.ndarray): An image read with OpenCV, represented as a numpy array with
            three channels in BGR format.
        points (np.ndarray): A numpy array of shape (n, 2) representing the (x, y)
            coordinates of n keypoints in the image. The array should have data type np.float32.
        title (str, optional): A string specifying the title of the window in which
            the image will be displayed. Defaults to "figure".
        save_path (str, optional): If specified, the image will be saved to this file
            path in addition to being displayed.

    Raises:
        AssertionError: If the inputs do not meet the specified requirements.
    """

    # Check image type and number of channels
    assert isinstance(image, np.ndarray), "Image must be a numpy array"
    assert (
        len(image.shape) == 3 and image.shape[2] == 3
    ), "Image must have three channels"
    # Check points type and shape
    assert isinstance(points, np.ndarray), "Points must be a numpy array"
    assert points.shape[1] == 2, "Points must be a nx2 numpy array"
    # Check title type
    assert isinstance(title, str), "Title must be a string"

    img_kpts = cv2.drawKeypoints(
        image,
        cv2.KeyPoint.convert(points),
        image,
        (255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img_kpts)

    if save_path:
        cv2.imwrite(save_path, img_kpts)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_features(
    image: np.ndarray,
    features: Features,
    title: str = None,
    ax=None,
    save_path: Union[str, Path] = None,
    hide_fig: bool = False,
    **kwargs,
) -> None:
    """Wrapper around plot_points to work if the input is a Features object"""
    xy = features.to_numpy()["kpts"]
    fig = plot_points(
        image,
        points=xy,
        title=title,
        ax=ax,
        save_path=save_path,
        hide_fig=hide_fig,
        **kwargs,
    )


def plot_feature(
    image: np.ndarray,
    feature: Feature,
    title: str = None,
    ax=None,
    save_path: Union[str, Path] = None,
    hide_fig: bool = False,
    zoom_to_feature: bool = False,
    window_size: int = 50,
    **kwargs,
) -> None:
    """Wrapper around plot_points to work if the input is a single Feature object"""
    xy = feature.xy
    ax = plot_points(
        image,
        points=xy,
        title=title,
        ax=ax,
        save_path=save_path,
        hide_fig=hide_fig,
        zoom_to_features=zoom_to_feature,
        window_size=window_size,
        **kwargs,
    )


def plot_projections(
    points3d, camera: Camera, image, title: str = None, ax=None
) -> None:
    """
    Project 3D points to a camera and plot their projections on an image.

    Args:
        points3d: numpy.ndarray
            Array of shape (n, 3) containing 3D points in object space.
        camera: Camera object
            Camera object containing exterior and interior orientation.
        image: numpy.ndarray
            Image with BGR channels in OpenCV standard.
        title: str, optional
            Title of the axes of the plot.
        ax: matplotlib axes, optional
            Axis in which to make the plot. If nothing is given, the function creates
            a new figure and axes.

    Returns:
        None
    """

    if ax is None:
        fig, ax = plt.subplots()

    projections = project_points(points3d, camera)
    plot_features(image, projections, title, ax)


def plot_projection_error(
    projections,
    projection_error,
    image,
    title: str = None,
    ax=None,
    convert_BRG2RGB=True,
) -> None:
    """
    Plot the projection error of 3D points on an image.

    Args:
        projections: numpy.ndarray
            Array of shape (n, 2) containing the projections of 3D points on an image.
        projection_error: numpy.ndarray
            Array of shape (n,) containing the projection error for each point.
        image: numpy.ndarray
            Image with BGR channels in OpenCV standard.
        title: str, optional
            Title of the plot.
        ax: matplotlib axes, optional
            Axis in which to make the plot. If nothing is given, the function creates
            a new figure and axes.
        convert_BRG2RGB: bool, optional
            Whether to convert the image from BGR to RGB.

    Returns:
        None
    """

    if convert_BRG2RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    viridis = cm.get_cmap("viridis", 8)
    norm = Colors.Normalize(vmin=projection_error.min(), vmax=projection_error.max())
    cmap = viridis(norm(projection_error))

    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.imshow(image)
    scatter = ax.scatter(
        projections[:, 0],
        projections[:, 1],
        s=10,
        c=cmap,
        marker="o",
        # alpha=0.5, edgecolors='k',
    )
    ax.set_title(title)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Reprojection error in y")


def draw_epip_lines(img0, img1, lines, pts0, pts1, fast_viz=True):
    """
    Draw epipolar lines on two images for corresponding points.

    Args:
        img0: numpy.ndarray
            Image on which we draw the epilines for the points in img1.
        img1: numpy.ndarray
            Image on which we draw the epilines for the points in img0.
        lines: numpy.ndarray
            Array of shape (n, 3) containing epipolar lines.
        pts0: numpy.ndarray
            Array of shape (n, 2) containing points in img0.
        pts1: numpy.ndarray
            Array of shape (n, 2) containing points in img1.
        fast_viz: bool, optional
            Whether to use a fast visualization method.

    Returns:
        Two numpy.ndarrays containing the input images with epipolar lines and points drawn on them.
    """
    r, c, _ = img0.shape
    if not fast_viz:
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # @TODO: implement visualization in matplotlib
    for r, pt0, pt1 in zip(lines, pts0, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img0 = cv2.line(img0, (x0, y0), (x1, y1), color, 1)
        img0 = cv2.circle(img0, tuple(pt0.astype(int)), 5, color, -1)
        img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
        # img0 = cv2.drawMarker(img0,tuple(pt0.astype(int)),color,cv2.MARKER_CROSS,3)
        # img1 = cv2.drawMarker(img1,tuple(pt1.astype(int)),color,cv2.MARKER_CROSS,3)
    return img0, img1


""" Misc functions"""


def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))


""" Visualization of 3D point clouds"""


def display_point_cloud(
    point_clouds: Union[PointCloud, List[PointCloud]],
    cameras: List[Camera] = None,
    viz_rs: bool = True,
    win_name: str = "Point cloud",
    plot_scale: int = 5,
    visible: bool = True,
) -> None:
    """Display a O3D point cloud
    Parameters
    ----------
    pcd : PointCloud obejct
    cameras : List of Camera objects (default = None)
        List of Camera objects, used to visualize the location
        and orientation of the cameras in the plot.
        If None is given, only the point cloud is plotted.
    visible : set to false to run headless
    Returns
    -------
    None.
    """

    if isinstance(point_clouds, PointCloud):
        plt_objs = [point_clouds.pcd]
    elif isinstance(point_clouds, List):
        plt_objs = [x.pcd for x in point_clouds]

    if cameras is not None:
        num_cams = len(cameras)

        if num_cams < 3:
            cam_colors = [[1, 0, 0], [0, 0, 1]]
        else:
            cam_colors = np.full((num_cams, 3), [1, 0, 0])
            cam_colors[2:4] = [[0, 0, 1], [0, 0, 1]]

        for i, cam in enumerate(cameras):
            plt_objs.append(
                make_camera_pyramid(
                    cam,
                    color=cam_colors[i],
                    focal_len_scaled=plot_scale,
                )
            )
    if viz_rs:
        plt_objs.append(make_viz_sdr(scale=plot_scale * 2))

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=win_name,
        width=1280,
        height=720,
        left=300,
        top=200,
        visible=visible,
    )
    for x in plt_objs:
        vis.add_geometry(x)
    vis.run()
    vis.destroy_window()


def display_pc_inliers(cloud, ind):
    """Display a O3D point cloud, separating inliers from outliers
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
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    logging.info("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def make_camera_pyramid(
    camera: Camera, color=[1.0, 0.0, 0.0], focal_len_scaled=5, aspect_ratio=0.3
):
    """
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

    """
    # Check if camera pose is available, otherwise build it.
    if camera.pose is None:
        camera.Rt_to_extrinsics()
        camera.extrinsics_to_pose()
    vertexes = pose2pyramid(camera.pose, focal_len_scaled, aspect_ratio)

    # Build Open3D camera object
    vertex_pcd = o3d.geometry.PointCloud()
    vertex_pcd.points = o3d.utility.Vector3dVector()

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    color = np.array(color, dtype="float32")
    colors = [color for i in range(len(lines))]

    cam_view_obj = o3d.geometry.LineSet()
    cam_view_obj.lines = o3d.utility.Vector2iVector(lines)
    cam_view_obj.colors = o3d.utility.Vector3dVector(colors)
    cam_view_obj.points = o3d.utility.Vector3dVector(vertexes)

    return cam_view_obj


def pose2pyramid(camera_pose, focal_len_scaled=5, aspect_ratio=0.3):
    # TODO: add description
    """Function inspired from https://github.com/demul/extrinsic2pyramid

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

    """
    vertex_std = np.array(
        [
            [0, 0, 0, 1],
            [
                focal_len_scaled * aspect_ratio,
                -focal_len_scaled * aspect_ratio,
                focal_len_scaled,
                1,
            ],
            [
                focal_len_scaled * aspect_ratio,
                focal_len_scaled * aspect_ratio,
                focal_len_scaled,
                1,
            ],
            [
                -focal_len_scaled * aspect_ratio,
                focal_len_scaled * aspect_ratio,
                focal_len_scaled,
                1,
            ],
            [
                -focal_len_scaled * aspect_ratio,
                -focal_len_scaled * aspect_ratio,
                focal_len_scaled,
                1,
            ],
        ],
        dtype=np.float32,
    )
    vertex_transformed = vertex_std @ camera_pose.T

    return vertex_transformed[:, 0:3]


def make_viz_sdr(scale=5):
    """
    Parameters
    ----------
    scale : int
        scale for plotting the axes vectors
    Returns
    -------
    rs_obj : o3d geometry
        Open3D geometry object containing the reference system visualization symbol
        Colors are R, G, B respectively for X, Y, Z axis
    """
    vertexes = scale * np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    colors = np.array(
        [[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0]], dtype="float32"
    )
    lines = [[0, 1], [0, 2], [0, 3]]

    rs_obj = o3d.geometry.LineSet()
    rs_obj.lines = o3d.utility.Vector2iVector(lines)
    rs_obj.colors = o3d.utility.Vector3dVector(colors)
    rs_obj.points = o3d.utility.Vector3dVector(vertexes)

    return rs_obj


""" Various plot for paper """


def make_focal_length_variation_plot(
    focals,
    save_path: Union[str, Path] = None,
) -> None:
    epoches_2_process = range(len(focals[0]))
    n_cams = len(focals)
    fig, ax = plt.subplots(1, n_cams)
    for s_id in range(n_cams):
        ax[s_id].plot(epoches_2_process, focals[s_id], "o")
        ax[s_id].grid(visible=True, which="both")
        ax[s_id].set_xlabel("Epoch")
        ax[s_id].set_ylabel("Focal lenght [px]")
    if save_path is None:
        fig.show()
    else:
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(save_path, dpi=100)


def make_camera_angles_plot(
    cameras,
    save_path: Union[str, Path] = None,
    baseline_epoch: int = None,
    current_epoch: int = None,
):

    cameras_plt = deepcopy(cameras)

    if baseline_epoch is not None:
        ep0 = baseline_epoch
    else:
        ep0 = 0

    if current_epoch is not None:
        epoches = range(ep0, current_epoch + 1)
    else:
        epoches = range(ep0, len(cameras_plt))

    cam_keys = list(cameras_plt[ep0].keys())
    angles_keys = ["omega", "phi", "kappa"]
    angles = dict.fromkeys(cam_keys)
    for key in cam_keys:
        angles[key] = {}
        for angle in angles_keys:
            angles[key][angle] = {}

    for ep, cam_dict in cameras_plt.items():
        if ep in epoches:
            for cam_key, cam in cam_dict.items():
                omega = cam.euler_angles[0]
                phi = cam.euler_angles[1]
                kappa = cam.euler_angles[2]
                angles[cam_key]["omega"][ep] = omega
                angles[cam_key]["phi"][ep] = phi
                angles[cam_key]["kappa"][ep] = kappa

                if baseline_epoch is not None:
                    omega0 = angles[cam_key]["omega"][ep0]
                    phi0 = angles[cam_key]["phi"][ep0]
                    kappa0 = angles[cam_key]["kappa"][ep0]
                    angles[cam_key]["omega"][ep] -= omega0
                    angles[cam_key]["phi"][ep] -= phi0
                    angles[cam_key]["kappa"][ep] -= kappa0

    fig, ax = plt.subplots(3, 2)
    if baseline_epoch is not None:
        fig.suptitle(
            f"Attitude angles of the two cameras_plt - differences with respect to epoch {baseline_epoch}"
        )
    else:
        fig.suptitle(f"Attitude angles of the two cameras")
    for i, cam_key in enumerate(cam_keys):
        ax[0, i].plot(epoches, list(angles[cam_key]["omega"].values()), "o")
        ax[0, i].grid(visible=True, which="both")
        ax[0, i].set_xlabel("Epoch")
        ax[0, i].set_ylabel(f"Omega [deg]")
        ax[0, i].set_title(f"Camera {cam_key}")

        ax[1, i].plot(epoches, list(angles[cam_key]["phi"].values()), "o")
        ax[1, i].grid(visible=True, which="both")
        ax[1, i].set_xlabel("Epoch")
        ax[1, i].set_ylabel(f"Phi [deg]")
        ax[1, i].set_title(f"Camera {cam_key}")

        ax[2, i].plot(epoches, list(angles[cam_key]["kappa"].values()), "o")
        ax[2, i].grid(visible=True, which="both")
        ax[2, i].set_xlabel("Epoch")
        ax[2, i].set_ylabel(f"Kappa [deg]")
        ax[2, i].set_title(f"Camera {cam_key}")
    fig.tight_layout(pad=0.05)

    if save_path is None:
        fig.show()
    else:
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(save_path, dpi=100)


""" Other skatched functions to be implemented"""
# Darw features
# img0 = images[1][0]
# pt0 = features[1][epoch].kpts_to_numpy()
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


if __name__ == "__main__":

    camera_extrinsics = np.eye(4)
    K = np.array([[4000, 0, 2000], [0, 4000, 3000], [0, 0, 1]])
    cam = Camera(6000, 4000, K, extrinsics=camera_extrinsics)

    pyr = make_camera_pyramid(
        cam, color=[1.0, 0.0, 0.0], focal_len_scaled=2, aspect_ratio=0.4
    )
    o3d.visualization.draw_geometries([pyr])
    o3d.io.write_line_set("camera_sym.ply", pyr, write_ascii=True)

    print("done")
