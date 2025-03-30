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
