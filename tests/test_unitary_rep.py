import pickle

import numpy as np
import pennylane as qml
import pytest

from graddft_qnn.unitary_rep import O_h, is_zero_matrix_combination


@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_x_axis_180_permutation_matrix(size):
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
    assert np.allclose(
        O_h._180_deg_x_rot(size, pauli_word=False),
        qml.matrix(O_h._180_deg_x_rot_sparse(size, pauli_word=True)),
    )


@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_x_axis_90_permutation_matrix(size):
    expected_array = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
    )
    assert np.allclose(O_h._90_deg_x_rot(2), expected_array)
    assert np.allclose(qml.matrix(qml.I(0) @ qml.X(1) @ qml.X(2)), expected_array)
    assert np.allclose(
        O_h._90_deg_x_rot(size, pauli_word=False),
        qml.matrix(O_h._90_deg_x_rot_sparse(size, pauli_word=True)),
    )


@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_x_axis_180_permutation_matrix_qubits(size):
    prod = O_h._180_deg_x_rot(size, True)
    assert np.allclose(O_h._180_deg_x_rot(size), qml.matrix(prod))


@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_y_axis_180_permutation_matrix(size):
    prod = O_h._180_deg_y_rot(size, True)
    assert np.allclose(O_h._180_deg_y_rot(size), qml.matrix(prod))


@pytest.mark.parametrize("size", [2, 4, 8, 16])
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
    assert np.allclose(O_h.yz_reflection(2), expected_array)


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
    assert np.allclose(O_h.xz_reflection(2), expected_array)


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
    assert np.allclose(O_h.xy_reflection(2), expected_array)


def test_y_neg_eq_z_reflect():
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
    assert np.allclose(O_h.y_eq_neg_z_rot(2), expected_array)


def test_y_eq_z_reflect():
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
    assert np.allclose(O_h.y_eq_z_rot(2), expected_array)
    assert np.allclose(O_h.y_eq_z_rot(4), O_h._90_deg_x_rot(4) @ O_h._180_deg_z_rot(4))


def test_inversion():
    expected_array = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert np.allclose(O_h.inversion(2), expected_array)


def test_y_equal_z_reflection():
    expected_array = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ]
    )
    assert np.allclose(O_h.y_equal_z_reflection(2), expected_array)


def test_y_equal_neg_z_reflection():
    expected_array = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    assert np.allclose(O_h.y_equal_neg_z_reflection(2), expected_array)


def test_is_zero_matrix_combination():
    exp1 = qml.X(0) @ qml.X(1) @ qml.Y(2)
    exp1_1 = qml.X(0) @ qml.X(1) @ qml.X(2)
    exp2_dense = O_h._180_deg_x_rot(2, pauli_word=True)
    exp2_sparse = O_h._180_deg_x_rot_sparse(2, pauli_word=True)
    assert not is_zero_matrix_combination(
        exp1_1 + exp2_dense @ exp1_1 @ qml.adjoint(exp2_dense)
    )
    assert not is_zero_matrix_combination(
        exp1_1 + exp2_sparse @ exp1_1 @ qml.adjoint(exp2_sparse)
    )
    assert is_zero_matrix_combination(
        exp1 + exp2_dense @ exp1 @ qml.adjoint(exp2_dense)
    )
    assert is_zero_matrix_combination(
        exp1 + exp2_sparse @ exp1 @ qml.adjoint(exp2_sparse)
    )

    exp1_3 = qml.X(0) @ qml.X(1) @ qml.X(2) @ qml.X(3) @ qml.Y(4) @ qml.Y(5)
    exp2_1_sparse = O_h._180_deg_x_rot_sparse(4, True)
    exp2_1 = O_h._180_deg_x_rot(4, True)
    assert not is_zero_matrix_combination(
        exp1_3 + exp2_1 @ exp1_3 @ qml.adjoint(exp2_1)
    )
    assert not is_zero_matrix_combination(
        exp1_3 + exp2_1_sparse @ exp1_3 @ qml.adjoint(exp2_1_sparse)
    )


def test_sparse_vs_dense():
    with open("tests/ansatz_6_180_deg_x_rot.pkl", "rb") as f:
        ansatz_dense = pickle.load(f)
    with open("tests/ansatz_6_180_deg_x_rot_sparse.pkl", "rb") as f:
        ansatz_sparse = pickle.load(f)
    for i, j in zip(ansatz_dense, ansatz_sparse):
        assert np.allclose(qml.matrix(i), qml.matrix(j))
