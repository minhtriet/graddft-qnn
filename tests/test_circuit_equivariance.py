import numpy
import pennylane as qml
import pennylane.numpy as np
import pytest

from graddft_qnn import custom_gates
from graddft_qnn.unitary_rep import O_h

dev = qml.device("default.qubit", wires=3)
dev_1 = qml.device("default.qubit", wires=6)


@qml.qnode(dev)
def circuit_2(feature):
    qml.AmplitudeEmbedding(feature, wires=dev.wires, pad_with=0.0)
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


@qml.qnode(dev)
def circuit_2_with_pool(feature):
    qml.AmplitudeEmbedding(feature, wires=dev.wires, pad_with=0.0)
    qml.RX(1, 0)
    qml.RX(2, 1)
    qml.RX(3, 2)
    qml.MultiRZ(np.pi * 0.33, [0, 1, 2])
    # ====
    # Pool
    qml.ctrl(custom_gates.RXX(), control=0, control_values=[1], work_wires=[1, 2])

    return (
        qml.expval(qml.X(1)),
        qml.expval(qml.X(2)),
        qml.expval(qml.Z(1) @ qml.Z(2)),
    )

# run pyscf
#

@qml.qnode(dev_1)
def circuit_2_with_pool_6_wires(feature):
    qml.AmplitudeEmbedding(feature, wires=dev.wires, pad_with=0.0)
    qml.RX(1, 0)
    qml.RX(2, 1)
    qml.RX(3, 2)
    qml.MultiRZ(np.pi * 0.33, [0, 1, 2])
    qml.RX(1, 3)
    qml.RX(2, 4)
    qml.RX(3, 5)
    qml.MultiRZ(np.pi * 0.33, [3, 4, 5])
    # ====
    # Pool
    qml.ctrl(custom_gates.RXX(), control=0, control_values=[1], work_wires=[1, 2])
    qml.ctrl(custom_gates.RXX(), control=3, control_values=[1], work_wires=[4, 5])

    return (
        qml.expval(qml.X(1)),
        qml.expval(qml.X(2)),
        qml.expval(qml.Z(1) @ qml.Z(2)),
        qml.expval(qml.X(4)),
        qml.expval(qml.X(5)),
        qml.expval(qml.Z(4) @ qml.Z(5)),
    )

@pytest.mark.parametrize(
    "n_wires, circuit_func", [(3, circuit_2),
                              (3, circuit_2)]
)
def test_invariant(n_wires, circuit_func):
    feature = np.random.rand(2**n_wires)
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
