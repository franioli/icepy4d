import pytest
import numpy as np
from icepy4d.utils.transformations import (
    Rotrotranslation,
    convert_to_homogeneous,
    convert_from_homogeneous,
)


# tests for the Rotrotranslation class
def test_rotrotranslation_init():
    t_mat = np.eye(4)
    rotrotranslation = Rotrotranslation(t_mat)
    assert rotrotranslation.T.shape == (4, 4)


def test_rotrotranslation_apply_eye_transformation():
    t_mat = np.eye(4)
    rotrotranslation = Rotrotranslation(t_mat)
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]])
    expected_result = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = rotrotranslation.apply_transformation(x)
    assert np.allclose(result, expected_result)


# tests for the conversion functions
def test_convert_to_homogeneous():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    expected_result = np.array([[1, 2, 3], [4, 5, 6], [1, 1, 1]])
    result = convert_to_homogeneous(x)
    assert np.allclose(result, expected_result)


def test_convert_from_homogeneous():
    x = np.array([[1, 2, 3], [4, 5, 6], [1, 1, 1]])
    expected_result = np.array([[1, 2, 3], [4, 5, 6]])
    result = convert_from_homogeneous(x)
    assert np.allclose(result, expected_result)


if __name__ == "__main__":
    pytest.main([__file__])  # run all tests in this file
