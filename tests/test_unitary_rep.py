import numpy as np
import pennylane as qml
import pytest

from graddft_qnn.unitary_rep import O_h


def test_x_axis_180_permutation_matrix():
    expected_array = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ]
    )
    assert np.allclose(O_h._180_deg_x_rot(2), expected_array)
    assert np.allclose(qml.matrix(qml.I(0) @ qml.X(1) @ qml.X(2)), expected_array)


@pytest.mark.parametrize("size", [2, 4, 8, 16, 32])
def test_x_axis_180_permutation_matrix_qubits(size):
    prod = O_h._180_deg_x_rot(size, True)
    assert np.allclose(O_h._180_deg_x_rot(size), qml.matrix(prod))


@pytest.mark.parametrize("size", [2, 4, 8, 16, 32])
def test_y_axis_180_permutation_matrix(size):
    prod = O_h._180_deg_y_rot(size, True)
    assert np.allclose(O_h._180_deg_y_rot(size), qml.matrix(prod))


@pytest.mark.parametrize("size", [2, 4, 8, 16, 32])
def test_z_axis_180_permutation_matrix(size):
    prod = O_h._180_deg_z_rot(size, True)
    assert np.allclose(O_h._180_deg_z_rot(size), qml.matrix(prod))


def test_x_reflect():
    expected_array = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    assert np.allclose(O_h._x_eq_0_reflection(2), expected_array)


def test_y_reflect():
    expected_array = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
        ]
    )
    assert np.allclose(O_h._y_eq_0_reflection(2), expected_array)


def test_z_reflect():
    expected_array = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
    )
    assert np.allclose(O_h._z0_reflection(2), expected_array)


def test_y_eq_z_reflect():
    expected_array = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    assert np.allclose(O_h.y_eq_z_rot(2), expected_array)

def test_y_eq_neg_z_reflect():
    expected_array = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert np.allclose(O_h.y_eq_neg_z_rot(2), expected_array)
