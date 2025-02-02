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

# todo I suppose w would invent something like RX(a) \otimes RY(b) \otimes RZ(c). I suppose that a, b, c is learnable param?
# then doesn't make sense to add that to test code? Or we just add the param that we know should work
# todo how closely is it related to invent a new material?
# minimize hammming distance ?
@pytest.mark.parametrize(
    "feature,psi,theta,phi,group_matrix,expected",
    [
        (np.arange(1, 2**3 + 1), np.pi, 0, 0, x_rot_matrix, [6, 5, 8, 7, 2, 1, 4, 3]),
        (np.arange(1, 2**3 + 1), np.pi, 0, 0, z_rot_matrix, [4, 3, 2, 1, 8, 7, 6, 5]),
        # todo add more test here to cover D2h group
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
