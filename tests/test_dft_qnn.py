import numpy as np
import pennylane as qml

from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.unitary_rep import O_h


def test_twirling():
    # eq.60 in https://doi.org/10.1103/PRXQuantum.4.010328
    twirled = DFTQNN._twirling(qml.X(0) @ qml.I(1), [qml.SWAP([0, 1])])
    expected = 0.5 * (qml.X(0) + qml.X(1))
    assert np.allclose(qml.matrix(twirled[0]), qml.matrix(expected))

    # eq.65 in https://doi.org/10.1103/PRXQuantum.4.010328
    twirled = DFTQNN._twirling(qml.I(0) @ qml.Y(1), [qml.X(0) @ qml.X(1)])
    assert twirled is None


def test_gate_design():
    gate_gen = DFTQNN.gate_design(6, O_h.C2_group(4, pauli_word=True))
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
