import numpy as np
import pandas as pd
import argparse

" Transformation for Belvedere North-West terminus, from Local RS to WGS84-UTM32N "
BELV_LOC2UTM = np.array(
    [
        [0.706579327583, -0.70687371492, -0.00012600114, 416614.833],
        [0.706873714924, 0.706579267979, 0.000202054813, 5090932.706],
        [-0.00005382637, -0.00023195939, 0.999462246895, 1767.547],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
BELV_UTM2LOC = np.linalg.inv(BELV_LOC2UTM)


def parse_command_line():
    pass


def convert_to_homogeneous(x: np.ndarray) -> np.ndarray:
    """Converts a 2xn or 3xn vector of n points in euclidean coordinates to a 3xn or 4xn vector in homogeneous coordinates by adding a row of ones.

    Args:
        x (np.ndarray): A numpy array with shape 2xn or 3xn, representing the euclidean
            coordinates of n points.

    Returns:
        np.ndarray: A numpy array with shape 3xn or 4xn, representing the homogeneous
        coordinates of the same n points.
    """
    x = np.array(x)
    ndim, npts = x.shape
    if ndim != 2 and ndim != 3:
        print(
            "Error: wrong number of dimension of the input vector.\
              A number of dimensions (rows) of 2 or 3 is required."
        )
        return None
    x1 = np.concatenate((x, np.ones((1, npts), "float32")), axis=0)
    return x1


def convert_from_homogeneous(x: np.ndarray) -> np.ndarray:
    """
    Convert 3xn or 4xn vector of n points in homogeneous coordinates
    to a 2xn or 3xn vector in euclidean coordinates, by dividing by the
    homogeneous part of the vector (last row) and removing one dimension
    """
    x = np.array(x)
    ndim, npts = x.shape
    if ndim != 3 and ndim != 4:
        print(
            "Error: wrong number of dimension of the input vector.\
              A number of dimensions (rows) of 2 or 3 is required."
        )
        return None
    x1 = x[: ndim - 1, :] / x[ndim - 1, :]
    return x1


def apply_transformation(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Applies a 4x4 transformation matrix in homogeneous coordinates
    to a 4xn numpy array in homogeneous coordinates. If input points are not in
    homogeneous coordinates, it converts them using convert_to_homogeneous function.

    Args:
        T: A 4x4 numpy array representing the transformation matrix.
        x: A 4xn numpy array representing n points in homogeneous coordinates or a 3xn numpy array representing n points in euclidean coordinates .

    Returns:
        A 3xn numpy array in euclidean coordinates, which is the result of applying the transformation matrix to the input array.
    """
    assert T.shape == (4, 4), "Error: T must be a 4x4 numpy array"

    # Check if x is in homogeneous coordinates, and convert it if not
    if x.shape[0] == 3 and not np.allclose(x[-1, :], np.ones(x.shape[1])):
        x = convert_to_homogeneous(x)

    if x.shape[0] != 4:
        raise ValueError(
            "Error: x must be a 4xn numpy array in homogeneous coordinates or a 3xn numpy array in euclidean coordinates."
        )

    out = T @ x
    return convert_from_homogeneous(out)


def get_coordinates_from_df(
    df: pd.DataFrame,
    to_homogeneous: bool = False,
) -> np.ndarray:
    """Extracts the 3D coordinates of a point cloud from a pandas DataFrame.

    Args:
        df: A pandas DataFrame with columns "X", "Y" and "Z".
        to_homogeneous: A boolean indicating whether to convert the coordinates
            to homogeneous coordinates or not.

    Returns:
        A numpy array with shape mx3 or mx4, representing the 3D coordinates
        of a point cloud with m points, in euclidean or homogeneous coordinates.
    """
    xyz = df[["X", "Y", "Z"]].to_numpy()
    if to_homogeneous:
        xyz = np.concatenate((xyz, np.ones((xyz.shape[0], 1))), axis=1)
    return xyz


def print_vector(vector):
    for i in vector:
        print(f"{i.squeeze():05.3f}")


def belv_loc2utm(point: np.ndarray):
    return apply_transformation(BELV_LOC2UTM, point)


def belv_utm2loc(point: np.ndarray):
    return apply_transformation(BELV_UTM2LOC, point)


def write_mat_to_csv(mat: np.ndarray, fname: str, sep: str = " "):
    with open(fname, "w") as f:
        for row in mat:
            string = f"{sep}".join([f"{x}" for x in row])
            f.write(f"{string}\n")


if __name__ == "__main__":

    # P = [-1.8212855433302195, 219.2598752180853, 74.55843482880829]

    # F10_utm = apply_transformation(BELV_LOC2UTM, np.array(P).reshape(3, 1))
    # print_vector(F10_utm)

    # write_mat_to_csv(BELV_UTM2LOC, "BELV_UTM2LOC.csv")
    # write_mat_to_csv(BELV_LOC2UTM, "BELV_LOC2UTM.csv")

    fname = "scripts/rototranslation/targets_loc.txt"
    points = pd.read_csv(fname)

    coor = get_coordinates_from_df(points, to_homogeneous=True)

    out = apply_transformation(BELV_LOC2UTM, coor.T).T

    points_utm = points.copy()
    points_utm[["X", "Y", "Z"]] = out

    fout = "scripts/rototranslation/targets_utm.txt"
    points_utm.to_csv(fout)

# import numpy as np
# import pandas as pd


# class Transformation:
#     def __init__(self, matrix):
#         """
#         Creates a transformation object with a 4x4 numpy array representing
#         the transformation matrix.

#         Args:
#             matrix: A 4x4 numpy array representing the transformation matrix.
#         """
#         assert matrix.shape == (4, 4), "Error: matrix must be a 4x4 numpy array"
#         self.matrix = matrix

#     def apply(self, points, homogeneous=False):
#         """
#         Applies the transformation matrix to a numpy array of 3D points.

#         Args:
#             points: A numpy array with shape mx3 representing the 3D coordinates
#                 of m points in euclidean coordinates.
#             homogeneous: A boolean indicating whether to convert the coordinates
#                 to homogeneous coordinates before applying the transformation.

#         Returns:
#             A numpy array with shape mx3 representing the 3D coordinates of the
#             transformed points in euclidean coordinates.
#         """
#         if homogeneous:
#             points = self._to_homogeneous(points)
#         transformed_points = apply_transformation(self.matrix, points.T).T
#         return transformed_points[:, :3]

#     def _to_homogeneous(self, points):
#         """
#         Converts a numpy array of 3D points from euclidean to homogeneous coordinates.

#         Args:
#             points: A numpy array with shape mx3 representing the 3D coordinates
#                 of m points in euclidean coordinates.

#         Returns:
#             A numpy array with shape mx4 representing the 3D coordinates of the
#             same m points in homogeneous coordinates.
#         """
#         return convert_to_homogeneous(points.T).T

#     def _from_homogeneous(self, points):
#         """
#         Converts a numpy array of 3D points from homogeneous to euclidean coordinates.

#         Args:
#             points: A numpy array with shape mx4 representing the 3D coordinates
#                 of m points in homogeneous coordinates.

#         Returns:
#             A numpy array with shape mx3 representing the 3D coordinates of the
#             same m points in euclidean coordinates.
#         """
#         return convert_from_homogeneous(points.T).T

#     @classmethod
#     def from_csv(cls, filename):
#         """
#         Loads a transformation matrix from a CSV file.

#         Args:
#             filename: A string representing the path to the CSV file.

#         Returns:
#             A Transformation object with the matrix loaded from the CSV file.
#         """
#         matrix = np.loadtxt(filename)
#         return cls(matrix)

#     @classmethod
#     def from_df(cls, df, to_homogeneous=False):
#         """
#         Creates a Transformation object with a matrix obtained from a pandas DataFrame.

#         Args:
#             df: A pandas DataFrame with columns "X", "Y" and "Z".
#             to_homogeneous: A boolean indicating whether to convert the coordinates
#                 to homogeneous coordinates before creating the transformation matrix.

#         Returns:
#             A Transformation object with the matrix created from the pandas DataFrame.
#         """
#         points = get_coordinates_from_df(df, to_homogeneous)
#         if to_homogeneous:
#             matrix = np.linalg.lstsq(points[:, :3], points[:, 3])[0]
#             matrix = np.vstack((matrix, [0, 0,
