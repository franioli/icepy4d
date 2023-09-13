import numpy as np
import pandas as pd
import open3d as o3d

from pathlib import Path
from typing import Union


"Transformation for Belvedere North-West terminus, from Local RS to WGS84-UTM32N"
BELV_LOC2UTM = np.array(
    [
        [0.706579327583, -0.70687371492, -0.00012600114, 416614.833],
        [0.706873714924, 0.706579267979, 0.000202054813, 5090932.706],
        [-0.00005382637, -0.00023195939, 0.999462246895, 1767.547],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class Rotrotranslation:
    def __init__(self, t_mat: np.ndarray) -> None:
        """
        Initializes a Rotrotranslation object with a transformation matrix.

        Args:
            t_mat: A 4x4 numpy array representing a transformation matrix in homogeneous coordinates.

        Raises:
            ValueError: If the input matrix does not meet the required criteria.
        """
        assert isinstance(t_mat, np.ndarray) and t_mat.shape == (
            4,
            4,
        ), "Invalid input for the transformation matrix t_mat. It must be a 4x4 numpy array."

        # Check if the last row is [0, 0, 0, 1]
        if not np.allclose(t_mat[3], [0, 0, 0, 1]):
            raise ValueError(
                "The transformation matrix must have [0, 0, 0, 1] in the last row for homogeneous coordinates."
            )

        # Extract the rotation and translation components
        rotation_matrix = t_mat[:3, :3]

        # Check if the rotation matrix is orthogonal (inv(R) = R.T)
        if not np.allclose(
            np.linalg.inv(rotation_matrix), rotation_matrix.T, rtol=1e-2
        ):
            raise ValueError(
                "The rotation matrix in the transformation matrix is not orthogonal (inv(R) != R.T)."
            )

        self._t = t_mat

    @property
    def T(self):
        return self._t

    @property
    def T_inv(self):
        return np.linalg.inv(self._t)

    @classmethod
    def read_T_from_file(cls, file: Union[str, Path]):
        """
        Reads a transformation matrix from a text file and creates a Rotrotranslation object.

        Args:
            file: Path to the text file containing the transformation matrix.
                  The file should be structured as a 4x4 matrix with whitespace-separated values.

        Returns:
            A Rotrotranslation object with the specified transformation matrix.

        Raises:
            ValueError: If the file does not contain a valid 4x4 transformation matrix.
        """
        try:
            T = np.loadtxt(file)
        except Exception as e:
            raise ValueError(
                f"Error reading the transformation matrix from the file: {e}"
            )

        if not isinstance(T, np.ndarray) or T.shape != (4, 4):
            raise ValueError(
                "Invalid transformation matrix format. The file must contain a 4x4 matrix."
            )

        return cls(T)

    @classmethod
    def belvedere_loc2utm(cls):
        """
        Return a Rotrotranslation object for transforming from Belvedere local coordinates to UTM coordinates.
        """
        return cls(BELV_LOC2UTM)

    @classmethod
    def belvedere_utm2loc(cls):
        """
        Return a Rotrotranslation object for transforming from UTM coordinates to Belvedere local coordinates.
        """
        BELV_UTM2LOC = np.linalg.inv(BELV_LOC2UTM)
        return cls(BELV_UTM2LOC)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Applies a 4x4 transformation matrix in homogeneous coordinates
        to a Nx3 or Nx4 numpy array representing n points in euclidean or homogeneous coordinates, respectively.

        If input points are not in homogeneous coordinates (Nx3), it converts them to homogeneous coordinates (Nx4) using convert_to_homogeneous function.

        Args:
            x: A Nx3 numpy array representing n points in euclidean coordinates or a Nx4 numpy array representing n points in homogeneous coordinates.

        Returns:
            A Nx3 numpy array in euclidean coordinates, which is the result of applying the transformation matrix to the input array.
        """

        assert isinstance(x, np.ndarray), "x must be a numpy array."
        assert (
            x.shape[1] == 3 or x.shape[1] == 4
        ), "x must be a nx3 in euclidean coordinates or nx4 numpy array in homogeneous coordinates."

        # Transposoute the input array to be in the form 3xn or 4xn
        x = x.T

        # Check if x is in homogeneous coordinates, and convert it if not
        if x.shape[0] == 3 or np.allclose(x[-1, :], np.ones(x.shape[1])):
            x = convert_to_homogeneous(x)

        # Apply the transformation matrix
        out = self.T @ x

        # Convert the output to euclidean coordinates and transpose it to be in the form nx3
        out = convert_from_homogeneous(out).T

        return out

    def transform_inverse(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the inverse of a 4x4 transformation matrix in homogeneous coordinates
        to a 4xn numpy array in homogeneous coordinates. If input points are not in homogeneous coordinates, it converts them using convert_to_homogeneous function.

        Args:
            x: A 4xn numpy array representing n points in homogeneous coordinates or a 3xn numpy array representing n points in euclidean coordinates.

        Returns:
            A 3xn numpy array in euclidean coordinates, which is the result of applying the inverse transformation matrix to the input array.
        """
        assert isinstance(x, np.ndarray), "x must be a numpy array."
        assert (
            x.shape[1] == 3 or x.shape[1] == 4
        ), "x must be a 3xn numpy array in euclidean coordinates or a 4xn numpy array in homogeneous coordinates."

        # Transpose the input array to be in the form 4xn or 3xn
        x = x.T

        # Check if x is in homogeneous coordinates, and convert it if not
        if x.shape[0] == 3 and not np.allclose(x[-1, :], np.ones(x.shape[1])):
            x = convert_to_homogeneous(x)

        if x.shape[0] != 4:
            raise ValueError(
                "Error: x must be a 4xn numpy array in homogeneous coordinates or a 3xn numpy array in euclidean coordinates."
            )

        out = self.T_inv @ x
        return convert_from_homogeneous(out).T

    def transform_pcd(
        self,
        pcd_path: Union[str, Path],
        out_path: str = None,
        inverse: bool = False,
    ) -> o3d.geometry.PointCloud:
        """
        Transforms a point cloud in PCD format using the current transformation matrix and saves the transformed point cloud to a file.

        Args:
            pcd_path (Union[str, Path]): The path to the input PCD file containing the point cloud data.
            out_path (str, optional): The output filename for the transformed point cloud. If not provided, a default filename will be generated.

        Returns:
            o3d.geometry.PointCloud: The transformed point cloud as an Open3D PointCloud object.
        """
        pcd_path = Path(pcd_path)

        # Read the point cloud from the input PCD file
        x = self.read_pcd(pcd_path)

        # Apply the transformation
        if not inverse:
            x_transf = self.transform(x)
        else:
            x_transf = self.transform_inverse(x)

        # Convert the transformed point cloud to a PCD object
        pcd_out = self.convert_points_to_pcd(x_transf)

        # Save the transformed point cloud to the specified file
        if out_path is not None:
            self.write_pcd(out_path, pcd_out)

        return pcd_out

    def read_pcd(self, fname: Union[str, Path]) -> np.ndarray:
        """
        Reads a point cloud from a PCD file using Open3D.

        Args:
            fname (Union[str, Path]): The path to the input PCD file.

        Returns:
            np.ndarray: A numpy array containing the point cloud data in the shape (N, 3), where N is the number of points in the point cloud.

        Raises:
            FileNotFoundError: If the input PCD file does not exist.
            ValueError: If there is an error reading the PCD file.
        """
        fname = Path(fname)
        try:
            pcd = o3d.io.read_point_cloud(str(fname))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The input point cloud file '{fname}' does not exist."
            )
        except Exception as e:
            raise ValueError(f"Error reading the point cloud file: {e}")

        return np.asarray(pcd.points)

    def convert_points_to_pcd(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Converts a numpy array of points to an Open3D PointCloud object.

        Args:
            points (np.ndarray): A numpy array containing the point cloud data in the shape (N, 3), where N is the number of points in the point cloud.

        Returns:
            o3d.geometry.PointCloud: The point cloud data as an Open3D PointCloud object.
        """
        assert points.shape[1] == 3, "points must be a Nx3 numpy array."

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd

    def write_pcd(
        self, out_path: Union[str, Path], pcd: o3d.geometry.PointCloud
    ) -> None:
        """
        Writes a point cloud to a PCD file using Open3D.

        Args:
            out_path (Union[str, Path]): The path to the output PCD file.
            pcd (o3d.geometry.PointCloud): An Open3D PointCloud object containing the point cloud data to be written.

        Returns:
            None

        Raises:
            ValueError: If there is an error writing the point cloud to the PCD file.
        """
        try:
            o3d.io.write_point_cloud(str(out_path), pcd)
        except Exception as e:
            raise ValueError(f"Error writing the point cloud to the PCD file: {e}")

    def write_T_mat_to_csv(self, fname: str, sep: str = " "):
        with open(fname, "w") as f:
            for row in self.T:
                string = f"{sep}".join([f"{x}" for x in row])
                f.write(f"{string}\n")


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


if __name__ == "__main__":
    belv_rotra = Rotrotranslation.belvedere_loc2utm()
    print(belv_rotra.T)

    rotra_from_file = Rotrotranslation.read_T_from_file(
        "scripts/rototranslation/BELV_LOC2UTM.txt"
    )
    print(rotra_from_file.T)

    # Apply transformation to a point
    fname = "data/targets/target_world.csv"
    target_loc = pd.read_csv(fname)
    points = target_loc[["X", "Y", "Z"]].to_numpy()
    points_utm = belv_rotra.transform(points)
    points_loc_t = belv_rotra.transform_inverse(points_utm)
    assert np.allclose(points, points_loc_t)

    # targets_utm = pd.DataFrame(columns=["label", "E", "N", "h"])
    # targets_utm["label"] = target_loc["label"]
    # targets_utm.to_csv(out_name, sep=",", index=False)

    # Work with point clouds
    pcd_path = "res/monthly_pcd/belvedere2021_densaMedium_lingua_50cm_utm.ply"
    out_path = "res/monthly_pcd/belvedere2021_densaMedium_lingua_50cm_loc.ply"
    pcd_transf = belv_rotra.transform_pcd(pcd_path, out_path=out_path)

    print("done")
