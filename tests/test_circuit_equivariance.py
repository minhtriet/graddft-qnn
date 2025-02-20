import numpy
import pennylane as qml
import pennylane.numpy as np

from graddft_qnn.unitary_rep import O_h
from  graddft_qnn import custom_gates
import pytest

dev = qml.device("default.qubit", wires=3)


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
    qml.ctrl(custom_gates.RXX(), control=0, control_values=[1], work_wires=[1,2])

    return (
        qml.expval(qml.X(1)),
        qml.expval(qml.X(2)),
        qml.expval(qml.Z(1) @ qml.Z(2)),
    )

# @pytest.mark.parametrize("circuit_func", [circuit_2])
@pytest.mark.parametrize("circuit_func", [circuit_2, circuit_2_with_pool])
def test_invariant(circuit_func):
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
