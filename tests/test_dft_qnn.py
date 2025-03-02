import numpy as np
import pennylane as qml
import pytest

from graddft_qnn.dft_qnn import DFTQNN


def fixed_circuit(feature, psi, theta, phi):
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(feature, psi, theta, phi):
        qml.AmplitudeEmbedding(feature, wires=dev.wires, normalize=True)
        for i in dev.wires[::3]:
            DFTQNN.U_O3(psi, theta, phi, wires=range(i, i + 3))
        return qml.probs()

    return circuit(feature, psi, theta, phi)


x_rot_matrix = np.array(
    [
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
    ]
)
y_rot_matrix = np.array([])
z_rot_matrix = np.array(
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
x_reflect_matrix = []
y_reflect_matrix = []
z_reflect_matrix = []


@pytest.mark.parametrize(
    "feature,psi,theta,phi,group_matrix,expected",
    [
        (np.arange(1, 2**3 + 1), np.pi, 0, 0, x_rot_matrix, [6, 5, 8, 7, 2, 1, 4, 3]),
        (np.arange(1, 2**3 + 1), np.pi, 0, 0, z_rot_matrix, [4, 3, 2, 1, 8, 7, 6, 5]),
    ],
)
def test_UO3_gate(feature, psi, theta, phi, group_matrix, expected):
    """
    Testing the equivariance of a quantum circuit
    """
    f_x = fixed_circuit(feature, psi, theta, phi)
    f_x_rot = fixed_circuit(expected, 0, 0, 0)
    rot_f_x = group_matrix @ f_x
    assert np.allclose(f_x_rot, rot_f_x, atol=1e-6)


def test_twirling():
    X_1 = qml.matrix(qml.X(0) @ qml.I(1))
    X_2 = qml.matrix(qml.I(0) @ qml.X(1))
    received = DFTQNN.twirling_(X_1, [qml.SWAP.compute_matrix()])
    expected = 0.5 * (X_1 + X_2)
    assert np.allclose(received, expected)

    Y_1 = qml.matrix(qml.Y(0) @ qml.I(1))
    received = DFTQNN.twirling_(Y_1, [qml.matrix(qml.X(0) @ qml.X(1))])
    expected = np.zeros(4)
    assert np.allclose(received, expected)
