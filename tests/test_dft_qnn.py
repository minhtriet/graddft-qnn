import functools

import numpy as np
import pennylane as qml
import pytest

from graddft_qnn.custom_gates import words
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.unitary_rep import O_h


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
    sentence = ["X"] * 6
    sentence_matrix = [words[x] for x in sentence]
    matrix = functools.reduce(np.kron, sentence_matrix)
    size = np.cbrt(matrix.shape[0])
    assert size.is_integer()
    assert np.allclose(matrix, DFTQNN.twirling_(matrix, O_h.C2_group(int(size))))

    sentence = ["X", "X", "X", "X", "I", "Z"]
    sentence_matrix = [words[x] for x in sentence]
    matrix = functools.reduce(np.kron, sentence_matrix)
    assert DFTQNN.twirling_(matrix, O_h.C2_group(int(size))) is None


def test_gate_design():
    gate_gen = DFTQNN.gate_design(6, O_h.C2_group(4))
    gate_gen = ["".join(g) for g in gate_gen]
    assert gate_gen == [
        "XXXXXX",
        "XXXXXI",
        "XXXXYY",
        "XXXXYZ",
        "XXXXZY",
        "XXXXZZ",
        "XXXXIX",
        "XXXXII",
        "XXXIXX",
        "XXXIXI",
        "XXXIYY",
        "XXXIYZ",
        "XXXIZY",
        "XXXIZZ",
        "XXXIIX",
        "XXXIII",
        "XXYYXX",
        "XXYYXI",
        "XXYYYY",
        "XXYYYZ",
        "XXYYZY",
        "XXYYZZ",
        "XXYYIX",
        "XXYYII",
        "XXYZXX",
        "XXYZXI",
        "XXYZYY",
        "XXYZYZ",
        "XXYZZY",
        "XXYZZZ",
        "XXYZIX",
        "XXYZII",
        "XXZYXX",
        "XXZYXI",
        "XXZYYY",
        "XXZYYZ",
        "XXZYZY",
        "XXZYZZ",
        "XXZYIX",
        "XXZYII",
        "XXZZXX",
        "XXZZXI",
        "XXZZYY",
        "XXZZYZ",
        "XXZZZY",
        "XXZZZZ",
        "XXZZIX",
        "XXZZII",
        "XXIXXX",
        "XXIXXI",
        "XXIXYY",
        "XXIXYZ",
        "XXIXZY",
        "XXIXZZ",
        "XXIXIX",
        "XXIXII",
        "XXIIXX",
        "XXIIXI",
        "XXIIYY",
        "XXIIYZ",
        "XXIIZY",
        "XXIIZZ",
        "XXIIIX",
        "XXIIII",
    ]
