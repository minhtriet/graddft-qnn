import unittest
import pennylane as qml
import pennylane.numpy as np
import numpy
from scipy.linalg import expm

from graddft_qnn.gates.ansatz import Ansatz
from graddft_qnn.unitary_rep import O_h

from graddft_qnn.dft_qnn import DFTQNN


class MyTestCase(unittest.TestCase):

    dev = qml.device("default.qubit", wires=3)

    @staticmethod
    @qml.qnode(dev)
    def circuit(feature, equivar_gate_matrix):
        qml.AmplitudeEmbedding(feature, wires=3, pad_with=0.0)
        for i in MyTestCase.dev.wires[::3]:
            qml.QubitUnitary(equivar_gate_matrix, wires=range(i, i + 3))
        return qml.expval(qml.Z(0)), qml.expval(qml.Z(1)), qml.expval(qml.Z(2))

    def test_something(self):
        # will calculate the coeff input without any dim reduction, might need to change that later.
        numpy.random.seed(42)
        feature = numpy.random.rand(8)
        unitary_reps = [O_h._180_deg_rot()]
        ansatz = Ansatz(np.pi, np.pi, np.pi, np.pi, [0,1,2])
        generator = DFTQNN.twirling(unitary_reps=unitary_reps, ansatz=qml.matrix(ansatz))
        equivar_gate_matrix = expm(generator)

        result = MyTestCase.circuit(feature, equivar_gate_matrix)
        assert result == 1


