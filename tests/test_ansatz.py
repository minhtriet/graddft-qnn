from functools import reduce

import numpy as np
import pennylane as qml

from graddft_qnn.ansatz import Ansatz


def test_wire_to_single_qubit_gates():
    ansatz = Ansatz()

    # Test for wire (0,)
    single_qubit_gate_0 = ansatz.wire_to_single_qubit_gates[(0,)]
    expected_matrix_0 = [
        reduce(np.kron, [qml.X.compute_matrix(), np.eye(2), np.eye(2)]),
        reduce(np.kron, [qml.Y.compute_matrix(), np.eye(2), np.eye(2)]),
        reduce(np.kron, [qml.Z.compute_matrix(), np.eye(2), np.eye(2)]),
    ]
    assert np.allclose(single_qubit_gate_0, expected_matrix_0)

    # Test for wire (1,)
    single_qubit_gate_1 = ansatz.wire_to_single_qubit_gates[(1,)]
    expected_matrix_1 = [
        reduce(np.kron, [np.eye(2), qml.X.compute_matrix(), np.eye(2)]),
        reduce(np.kron, [np.eye(2), qml.Y.compute_matrix(), np.eye(2)]),
        reduce(np.kron, [np.eye(2), qml.Z.compute_matrix(), np.eye(2)]),
    ]
    assert np.allclose(single_qubit_gate_1, expected_matrix_1)

    # Test for wire (2,)
    single_qubit_gate_2 = ansatz.wire_to_single_qubit_gates[(2,)]
    expected_matrix_2 = [
        reduce(np.kron, [np.eye(2), np.eye(2), qml.X.compute_matrix()]),
        reduce(np.kron, [np.eye(2), np.eye(2), qml.Y.compute_matrix()]),
        reduce(np.kron, [np.eye(2), np.eye(2), qml.Z.compute_matrix()]),
    ]
    assert np.allclose(single_qubit_gate_2, expected_matrix_2)
