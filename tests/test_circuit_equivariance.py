import unittest

import numpy
import pennylane as qml
import pennylane.numpy as np

from graddft_qnn.unitary_rep import O_h


class MyTestCase(unittest.TestCase):
    dev = qml.device("default.qubit", wires=3)

    def test_invariant_2(self):
        @qml.qnode(MyTestCase.dev)
        def circuit_2(feature):
            qml.AmplitudeEmbedding(feature, wires=MyTestCase.dev.wires, pad_with=0.0)
            qml.RX(1, 0)
            qml.RX(2, 1)
            qml.RX(3, 2)
            qml.MultiRZ(np.pi * 0.33, [0, 1, 2])
            return (
                qml.expval(qml.X(0)),
                qml.expval(qml.X(1)),
                qml.expval(qml.X(2)),
                qml.expval(qml.Z(0) @ qml.Z(1) @ qml.Z(2)),
            )

        feature = np.random.rand(8)
        rot_feature = O_h._180_deg_x_rot_matrix() @ feature
        lhs = circuit_2(feature)
        rhs = circuit_2(rot_feature)
        assert numpy.allclose(lhs, rhs)
        rot_feature = O_h._180_deg_y_rot_matrix() @ feature
        rhs = circuit_2(rot_feature)
        assert numpy.allclose(lhs, rhs)
        rot_feature = O_h._180_deg_z_rot_matrix() @ feature
        rhs = circuit_2(rot_feature)
        assert numpy.allclose(lhs, rhs)
