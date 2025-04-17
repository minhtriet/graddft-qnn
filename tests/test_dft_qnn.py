import numpy as np
import pennylane as qml

from graddft_qnn.dft_qnn import DFTQNN


def test_twirling():
    # eq.60 in https://doi.org/10.1103/PRXQuantum.4.010328
    twirled = DFTQNN._twirling(qml.X(0) @ qml.I(1), [qml.SWAP([0, 1])])
    expected = 0.5 * (qml.X(0) + qml.X(1))
    assert np.allclose(qml.matrix(twirled), qml.matrix(expected))

    # eq.65 in https://doi.org/10.1103/PRXQuantum.4.010328
    twirled = DFTQNN._twirling(qml.I(0) @ qml.Y(1), [qml.X(0) @ qml.X(1)])
    assert twirled is None
