import numpy as np
import pennylane as qml
import pytest

from graddft_qnn.dft_qnn import DFTQNN


@pytest.fixture
def circuit():
    dev = qml.device("default.qubit", wires=3)
    circuit = DFTQNN(dev)
    return circuit


def fixed_circuit(feature, psi, theta, phi):
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(feature, psi, theta, phi):
        qml.AmplitudeEmbedding(feature, wires=dev.wires, normalize=True)
        for i in dev.wires[::3]:
            DFTQNN.U_O3(psi, theta, phi, wires=range(i, i + 3))

        return qml.probs()

    return circuit(feature, psi, theta, phi)


@pytest.mark.parametrize(
    "feature,psi,theta,phi,expected",
    [(np.arange(1, 2**3 + 1), np.pi, 0, 0, [6, 5, 8, 7, 2, 1, 4, 3])
     # todo add more test here to cover D2h group
     ],
)
def test_quantum_circuit_with_embeddings(feature, psi, theta, phi, expected):
    """
    Testing the equivariance of a quantum circuit
    """
    f_x = fixed_circuit(feature, psi, theta, phi)
    f_x_rot = fixed_circuit(expected, 0, 0, 0)
    rot_f_x = (
        np.array(
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
        @ f_x
    )
    assert np.allclose(f_x_rot, rot_f_x, atol=1e-6)
