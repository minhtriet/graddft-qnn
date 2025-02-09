from graddft_qnn.ansatz import Ansatz
import pennylane as qml
import numpy as np

def test_wire_to_single_qubit_gates():
    ansatz = Ansatz()

    # Test for wire (0,)
    single_qubit_gate_0 = ansatz.wire_to_single_qubit_gates[(0,)]
    expected_matrix_0 = np.kron(qml.X.compute_matrix(), np.kron(qml.Y.compute_matrix(), qml.Z.compute_matrix()))
    assert np.allclose(single_qubit_gate_0, expected_matrix_0), "Test failed for wire (0,)"

    # Test for wire (1,)
    single_qubit_gate_1 = ansatz.wire_to_single_qubit_gates[(1,)]
    expected_matrix_1 = np.kron(np.eye(2), np.kron(qml.X.compute_matrix(), np.kron(qml.Y.compute_matrix(), qml.Z.compute_matrix())))
    assert np.allclose(single_qubit_gate_1, expected_matrix_1), "Test failed for wire (1,)"

    # Test for wire (2,)
    single_qubit_gate_2 = ansatz.wire_to_single_qubit_gates[(2,)]
    expected_matrix_2 = np.kron(np.eye(2), np.kron(np.eye(2), np.kron(qml.X.compute_matrix(), np.kron(qml.Y.compute_matrix(), qml.Z.compute_matrix()))))
    assert np.allclose(single_qubit_gate_2, expected_matrix_2), "Test failed for wire (2,)"

    # Test the case of all single qubit gates applied at once
    expected_all_gates_matrix = np.kron(qml.X.compute_matrix(), np.kron(qml.Y.compute_matrix(), qml.Z.compute_matrix()))
    all_single_qubit_gates = ansatz.wire_to_single_qubit_gates
    for wires, gate_matrix in all_single_qubit_gates.items():
        assert np.allclose(gate_matrix, expected_all_gates_matrix), f"Test failed for wire {wires}"
