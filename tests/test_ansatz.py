import numpy as np
import pennylane as qml

from graddft_qnn.ansatz import Ansatz


def test_wire_to_single_qubit_gates():
    ansatz = Ansatz()

    # Test for wire (0,)
    single_qubit_gate_0 = ansatz.wire_to_single_qubit_gates[(0,)]
    expected_matrix_0 = np.kron(
        qml.X.compute_matrix() @ qml.Y.compute_matrix() @ qml.Z.compute_matrix(),
        np.eye(4),
    )
    assert np.allclose(
        single_qubit_gate_0, expected_matrix_0
    ), "Test failed for wire (0,)"

    # Test for wire (1,)
    single_qubit_gate_1 = ansatz.wire_to_single_qubit_gates[(1,)]
    expected_matrix_1 = np.kron(
        np.kron(
            np.eye(2),
            qml.X.compute_matrix() @ qml.Y.compute_matrix() @ qml.Z.compute_matrix(),
        ),
        np.eye(2),
    )
    assert np.allclose(
        single_qubit_gate_1, expected_matrix_1
    ), "Test failed for wire (1,)"

    # Test for wire (2,)
    single_qubit_gate_2 = ansatz.wire_to_single_qubit_gates[(2,)]
    expected_matrix_2 = np.kron(
        np.eye(4),
        qml.X.compute_matrix() @ qml.Y.compute_matrix() @ qml.Z.compute_matrix()
    )
    assert np.allclose(
        single_qubit_gate_2, expected_matrix_2
    ), "Test failed for wire (2,)"
