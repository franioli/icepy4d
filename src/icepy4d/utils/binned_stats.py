import numpy as np
import cv2
import logging

from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime
from typing import List, Tuple
from scipy.stats import binned_statistic_2d, binned_statistic_dd


def bins_from_nodes(x_nodes: List, y_nodes: List) -> Tuple[List]:
    """Divides a 2D space into bins based on the x and y coordinates of the nodes.

    Args:
        x_nodes: A list of x coordinates for nodes.
        y_nodes: A list of y coordinates for nodes.

    Returns:
        A tuple of two lists, binx and biny, representing the boundaries of the bins.

    """
    step = x_nodes[1] - x_nodes[0]
    assert (
        step == y_nodes[1] - y_nodes[0]
    ), "Invalid input. Different step for x and y is not yet supported."
    binx = [x - step / 2 for x in x_nodes]
    binx.append(x_nodes[-1] + step / 2)
    biny = [x - step / 2 for x in y_nodes]
    biny.append(y_nodes[-1] + step / 2)
    return (binx, biny)


def print_points_in_grid(
    points: np.ndarray,
    values: np.ndarray = None,
    x_nodes: np.ndarray = None,
    y_nodes: np.ndarray = None,
    plot_point_id: bool = False,
) -> None:
    """Prints a scatter plot of points on a grid.

    Args:
        points: A numpy array of shape (n, 2) representing the x and y coordinates of n points.
        values: A numpy array of shape (n,) representing the values associated with each point.
        x_nodes: A numpy array of shape (m,) representing the x coordinates of m nodes.
        y_nodes: A numpy array of shape (k,) representing the y coordinates of k nodes.
        plot_point_id: A boolean indicating whether to label the points with their index.

    Returns:
        None
    """
    dx = 0.1
    binx, biny = bins_from_nodes(x_nodes, y_nodes)
    xx_bins, yy_bins = np.meshgrid(binx, biny)
    fig, ax = plt.subplots()
    sc = ax.scatter(points[:, 0], points[:, 1], c=values, label="Data points")
    ax.plot(xx_bins, yy_bins, "r")
    ax.plot(np.transpose(xx_bins), np.transpose(yy_bins), "r")
    if plot_point_id:
        for i, pt in enumerate(zip(points[:, 0], points[:, 1])):
            ax.text(pt[0] + dx, pt[1] + dx, str(i))
    ax.set_xlim(min(binx) - 4 * dx, max(binx) + 4 * dx)
    ax.set_ylim(min(biny) - 4 * dx, max(biny) + 4 * dx)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", "box")
    cbar = fig.colorbar(sc)
    cbar.set_label("values")
    plt.show()


def plot_binned_stats(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    values: np.ndarray,
    display_grid: bool = False,
    title: str = None,
) -> None:
    """
    Plots binned statistics from given x and y coordinates and their corresponding values.

    Args:
        x_nodes (np.ndarray): 2D array of x-coordinates (in meshgrid format).
        y_nodes (np.ndarray): 2D array of y-coordinates (in meshgrid format).
        values (np.ndarray): 2D array of values.
        display_grid (bool, optional): If True, display a grid of bins. Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If x_nodes, y_nodes, and values do not have the same length or are not 2D meshgrid arrays.

    Examples:
        >>> x_nodes = np.array([0, 1, 2, 3, 4])
        >>> y_nodes = np.array([0, 1, 2, 3, 4])
        >>> values = np.array([1, 2, 3, 4, 5])
        >>> xx_nodes, yy_nodes = np.meshgrid(x_nodes, y_nodes)
        >>> print_binned_stats(xx_nodes, yy_nodes, values)

    """
    dx = 0.1
    if (
        not isinstance(x_nodes, np.ndarray)
        or not isinstance(y_nodes, np.ndarray)
        or not isinstance(values, np.ndarray)
    ):
        raise ValueError("x_nodes, y_nodes, and values must be numpy arrays.")
    if not x_nodes.ndim == y_nodes.ndim == values.ndim == 2:
        raise ValueError("x_nodes, y_nodes, and values must be 2D meshgrid arrays.")
    if not x_nodes.shape == y_nodes.shape == values.shape:
        raise ValueError("x_nodes, y_nodes, and values must have the same shape.")

    xx_nodes, yy_nodes = x_nodes, y_nodes
    binx, biny = bins_from_nodes(xx_nodes[0, :], yy_nodes[:, 0])
    xx_bins, yy_bins = np.meshgrid(binx, biny)
    fig, ax = plt.subplots()
    sc = ax.scatter(xx_nodes, yy_nodes, c=values, label="Data points")
    if display_grid:
        ax.plot(xx_bins, yy_bins, "r")
        ax.plot(np.transpose(xx_bins), np.transpose(yy_bins), "r")
    ax.set_xlim(min(binx) - 4 * dx, max(binx) + 4 * dx)
    ax.set_ylim(min(biny) - 4 * dx, max(biny) + 4 * dx)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", "box")
    if title is not None:
        ax.set_title(title)
    cbar = fig.colorbar(sc)
    cbar.set_label("values")
    plt.show()


def compute_binned_stats2D(
    points_xy: np.ndarray,
    points_values: np.ndarray,
    statistic: str = "count",
    x_nodes: np.ndarray = None,
    y_nodes: np.ndarray = None,
    step: float = None,
    display_results: bool = False,
    title: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute binned statistics of the input point data.

    Args:
        points_xy (np.ndarray): Array of shape (n, 2) containing the x, y coordinates of the input points.
        points_values (np.ndarray): Array of shape (n,) containing the values of the input points.
        statistic (str, optional): The statistic to compute for each bin. Available options are:
            'count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var'. Refers to the documentation of scipy.stats.binned_statistic_2d for other options (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html). Defaults to 'count'.
        x_nodes (np.ndarray, optional): Array of shape (m,) containing the x coordinates of the bins. If None, x_nodes will be computed from the minimum and maximum x coordinates of the input points and the step size specified by the `step` parameter.
        y_nodes (np.ndarray, optional): Array of shape (m,) containing the y coordinates of the bins. If None, y_nodes will be computed from the minimum and maximum y coordinates of the input points and the step size specified by the `step` parameter.
        step (float, optional): The size of each bin in the x and y directions. Ignored if `x_nodes` and `y_nodes` are specified. If None, the default step size of 1 will be used.
        display_results (bool, optional): If True, plot the input points and the binned statistics.
        title (str, optional): The title of the plot if `display_results` is True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - xx_nodes (np.ndarray): Array of shape (m, m) containing the x coordinates of the bins.
        - yy_nodes (np.ndarray): Array of shape (m, m) containing the y coordinates of the bins.
        - statistic (np.ndarray): Array of shape (m, m) containing the binned statistics.

    Raises:
        AssertionError: If `x_nodes` or `y_nodes` are None and `step` is not specified.

    """

    if x_nodes is None or y_nodes is None:
        assert step is not None, "Missing 'step' value. Unable to compute nodes grid"
        x_nodes = np.arange(
            np.floor(min(points_xy[:, 0])), np.ceil(max(points_xy[:, 0])) + step, step
        )
        y_nodes = np.arange(
            np.floor(min(points_xy[:, 1])), np.ceil(max(points_xy[:, 1])) + step, step
        )
    binx, biny = bins_from_nodes(x_nodes, y_nodes)

    # print_points_in_grid(points_xy, points_values, x_nodes, y_nodes)
    ret = binned_statistic_2d(
        points_xy[:, 0].flatten(),
        points_xy[:, 1].flatten(),
        points_values.flatten(),
        statistic,
        bins=[binx, biny],
        expand_binnumbers=True,
    )
    xx_nodes, yy_nodes = np.meshgrid(x_nodes, y_nodes)

    if display_results:
        plot_binned_stats(xx_nodes, yy_nodes, ret.statistic.T, title=title)

    return (xx_nodes, yy_nodes, ret.statistic.T)


def bins_from_nodes3D(
    x_nodes: np.ndarray, y_nodes: np.ndarray, z_nodes: np.ndarray
) -> Tuple[List]:
    """Divides a 2D space into bins based on the x and y coordinates of the nodes.

    Args:
        x_nodes: A np.ndarray of x coordinates for nodes.
        y_nodes: A np.ndarray of y coordinates for nodes.
        z_nodes: A np.ndarray of z coordinates for nodes.

    Returns:
        A tuple of two lists, binx and biny, representing the boundaries of the bins.

    """
    step = x_nodes[1] - x_nodes[0]
    assert (
        step == y_nodes[1] - y_nodes[0]
    ), "Invalid input. Different step for x and y is not yet supported."
    binx = [v - step / 2 for v in x_nodes]
    binx.append(x_nodes[-1] + step / 2)
    biny = [v - step / 2 for v in y_nodes]
    biny.append(y_nodes[-1] + step / 2)
    binz = [v - step / 2 for v in z_nodes]
    binz.append(z_nodes[-1] + step / 2)
    return (binx, biny, binz)


def compute_binned_stats3D(
    points_xyz: np.ndarray,
    points_values: np.ndarray,
    statistic: str = "count",
    x_nodes: np.ndarray = None,
    y_nodes: np.ndarray = None,
    z_nodes: np.ndarray = None,
    step: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute binned statistics of the input point data.

    Args:
        points_xy (np.ndarray): Array of shape (n, 2) containing the x, y coordinates of the input points.
        points_values (np.ndarray): Array of shape (n,) containing the values of the input points.
        statistic (str, optional): The statistic to compute for each bin. Available options are:
            'count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var'. Refers to the documentation of scipy.stats.binned_statistic_2d for other options (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html). Defaults to 'count'.
        x_nodes (np.ndarray, optional): Array of shape (m,) containing the x coordinates of the bins. If None, x_nodes will be computed from the minimum and maximum x coordinates of the input points and the step size specified by the `step` parameter.
        y_nodes (np.ndarray, optional): Array of shape (m,) containing the y coordinates of the bins. If None, y_nodes will be computed from the minimum and maximum y coordinates of the input points and the step size specified by the `step` parameter.
        z_nodes (np.ndarray, optional): Array of shape (m,) containing the z coordinates of the bins. If None, z_nodes will be computed from the minimum and maximum z coordinates of the input points and the step size specified by the `step` parameter.
        step (float, optional): The size of each bin in the x and y directions. Ignored if `x_nodes` and `y_nodes` are specified. If None, the default step size of 1 will be used.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - xx_nodes (np.ndarray): Array of shape (m, m) containing the x coordinates of the bins.
        - yy_nodes (np.ndarray): Array of shape (m, m) containing the y coordinates of the bins.
        - statistic (np.ndarray): Array of shape (m, m) containing the binned statistics.

    Raises:
        AssertionError: If `x_nodes` or `y_nodes` are None and `step` is not specified.

    """

    if x_nodes is None or y_nodes is None:
        assert step is not None, "Missing 'step' value. Unable to compute nodes grid"
        x_nodes = np.arange(
            np.floor(min(points_xyz[:, 0])), np.ceil(max(points_xyz[:, 0])) + step, step
        )
        y_nodes = np.arange(
            np.floor(min(points_xyz[:, 1])), np.ceil(max(points_xyz[:, 1])) + step, step
        )
        z_nodes = np.arange(
            np.floor(min(points_xyz[:, 2])), np.ceil(max(points_xyz[:, 2])) + step, step
        )
    binx, biny, binz = bins_from_nodes3D(x_nodes, y_nodes, z_nodes)

    ret = binned_statistic_dd(
        points_xyz,
        points_values.flatten(),
        statistic,
        bins=[binx, biny, binz],
        expand_binnumbers=True,
    )
    xx_nodes, yy_nodes, zz_nodes = np.meshgrid(x_nodes, y_nodes, z_nodes)

    return (xx_nodes, yy_nodes, zz_nodes, ret.statistic)


# =========== Test with sample data =================
# npts = 20
# x = np.random.rand(npts) * 10
# y = np.random.rand(npts) * 10
# z = np.array(range(len(x)))
# step = 1

# x_nodes = np.arange(np.floor(min(x)), np.ceil(max(x)) + step, step)
# y_nodes = np.arange(np.floor(min(y)), np.ceil(max(y)) + step, step)
# xnode, ynode, magnitude_binned = compute_binned_stats(
#     np.stack((x, y), axis=0).T,
#     z.reshape(-1, 1),
#     statistic="median",
#     x_nodes=x_nodes,
#     y_nodes=y_nodes,
#     display_results=True,
#     title="magnitude",
# )
# binx, biny = bins_from_nodes(x_nodes, y_nodes)
# print_points_in_grid(
#     np.stack((x, y), axis=0).T, z.reshape(-1, 1), x_nodes, y_nodes, plot_point_id=True
# )
# ret = binned_statistic_2d(x, y, z, "median", bins=[binx, biny], expand_binnumbers=True)

# xx_nodes, yy_nodes = np.meshgrid(x_nodes, y_nodes)
# print_binned_stats(xx_nodes, yy_nodes, ret.statistic.T)


# =========== End test =================
