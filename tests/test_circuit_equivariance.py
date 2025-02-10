import functools
import unittest

import numpy
import pennylane as qml
import pennylane.numpy as np
from scipy.linalg import expm

from graddft_qnn.ansatz import Ansatz
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.unitary_rep import O_h


class MyTestCase(unittest.TestCase):
    dev = qml.device("default.qubit", wires=3)

    @staticmethod
    @qml.qnode(dev)
    def circuit(feature, equivar_gate_matrix):
        qml.AmplitudeEmbedding(feature, wires=MyTestCase.dev.wires, pad_with=0.0)
        for i in MyTestCase.dev.wires[::3]:
            qml.QubitUnitary(equivar_gate_matrix, wires=range(i, i + 3))
        return qml.expval(qml.Z(0)), qml.expval(qml.Z(1)), qml.expval(qml.Z(2))
        # todo make sure it works with twirling formular <X1><Z2>

    def test_invariant(self):
        numpy.random.seed(42)
        feature = numpy.random.rand(
            8
        )  # todo should also work with tensor, not just vanilla np
        rot_feature = O_h._180_deg_rot_matrix() @ feature

        unitary_reps = [O_h._180_deg_rot()]
        ansatz = Ansatz()

        circuit_rep : list[np.array] = []

        for wire, gates in ansatz.wire_to_single_qubit_gates.items():
            invariant_gates = []
            for gate in gates:
                generator = DFTQNN.twirling(unitary_reps=unitary_reps, ansatz=gate)
                invariant_gates.append(expm(-1j * 1 * generator))
            invariant_gates = functools.reduce(np.matmul, invariant_gates)

            pauli_invar_gate_sets = qml.pauli_decompose(invariant_gates, check_hermitian=False, hide_identity=True, pauli=True)
            circuit_rep.append(invariant_gates)

        circuit_rep = functools.reduce(np.kron, circuit_rep)

        for wire, gates in ansatz.wire_to_triple_qubit_gates.items():
            for gate in gates:
                generator = DFTQNN.twirling(unitary_reps=unitary_reps, ansatz=gate)
                circuit_rep = circuit_rep @ expm(-1j * 1 * generator)

        pauli_invar_gate_sets = qml.pauli_decompose(circuit_rep, check_hermitian=False, hide_identity=True)
        print(pauli_invar_gate_sets)
        # result = MyTestCase.circuit(feature, equivar_gate_matrix)
        # rot_result = MyTestCase.circuit(rot_feature, equivar_gate_matrix)
        # # same rotation matrix, but for 3d coordinates
        # r3 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        # # todo symmetrize gates one by one, create the matrix and decompose into pauli string
        # # f(R(x)) = f(x)

        # assert (result == rot_result).all()
