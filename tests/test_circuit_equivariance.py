import pathlib

import numpy
import pennylane as qml
import pennylane.numpy as np
import pytest
import yaml
from jax.random import PRNGKey

from graddft_qnn import custom_gates
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.io.ansatz_io import AnsatzIO
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


@qml.qnode(dev)
def circuit_3_wires(feature):
    qml.AmplitudeEmbedding(feature, wires=dev.wires, pad_with=0.0)
    custom_gates.U2(np.array(range(15)) / 15, 0)
    return custom_gates.U2_measurement(0)


@qml.qnode(dev_1)
def circuit_3_wires_6_wires(feature):
    qml.AmplitudeEmbedding(feature, wires=dev_1.wires, pad_with=0.0)
    custom_gates.U2_6_wires(np.array(range(15)) / 15, 0)
    return custom_gates.U2_6_wires_measurement(0)


@pytest.mark.parametrize(
    "n_wires, circuit_func",
    [
        # (3, circuit_2),
        #     (3, circuit_3_wires),
        (6, circuit_3_wires_6_wires)
    ],
)
def test_invariant(n_wires, circuit_func):
    feature = np.random.rand(2**n_wires)
    cube_dim = np.cbrt(2**n_wires)
    assert cube_dim.is_integer()
    cube_dim = int(cube_dim)
    rot_feature = O_h._180_deg_x_rot(cube_dim) @ feature
    lhs = circuit_func(feature)
    rhs = circuit_func(rot_feature)
    assert numpy.allclose(lhs, rhs)
    rot_feature = O_h._180_deg_y_rot(cube_dim) @ feature
    rhs = circuit_func(rot_feature)
    assert numpy.allclose(lhs, rhs)
    rot_feature = O_h._180_deg_z_rot(cube_dim) @ feature
    rhs = circuit_func(rot_feature)
    assert numpy.allclose(lhs, rhs)


@pytest.fixture
def _setup_device():
    with open("config.yaml") as file:
        data = yaml.safe_load(file)
        if "QBITS" not in data:
            raise KeyError("YAML file must contain 'QBITS' key")
        num_qubits = data["QBITS"]
    device = qml.device("default.qubit", wires=num_qubits)
    return device


def test_a_training_step(_setup_device):
    num_wires = len(_setup_device.wires)
    gates_indices = [7, 10, 14, 18, 20, 22, 38, 42, 57, 60]
    filename = pathlib.Path("tests") / "ansatz_6_qubits.txt"
    gates_gen = AnsatzIO.read_from_file(str(filename))
    measurement_expvals = [
        custom_gates.generate_operators(measurement) for measurement in gates_gen
    ]
    mock_coeff_inputs = np.random.rand(2**num_wires)
    dft_qnn = DFTQNN(_setup_device, gates_gen, measurement_expvals, gates_indices)

    rot_mock_coeff_inputs_x = (
        O_h._180_deg_x_rot(int(np.cbrt(2**num_wires))) @ mock_coeff_inputs
    )
    rot_mock_coeff_inputs_y = (
        O_h._180_deg_y_rot(int(np.cbrt(2**num_wires))) @ mock_coeff_inputs
    )
    rot_mock_coeff_inputs_z = (
        O_h._180_deg_z_rot(int(np.cbrt(2**num_wires))) @ mock_coeff_inputs
    )
    rot_mock_coeff_inputs_z_inputs_x = (
        O_h._180_deg_z_rot(int(np.cbrt(2**num_wires)))
        @ O_h._180_deg_x_rot(int(np.cbrt(2**num_wires)))
        @ mock_coeff_inputs
    )

    key = PRNGKey(42)

    parameters = dft_qnn.init(key, mock_coeff_inputs)

    result = dft_qnn.apply(parameters, mock_coeff_inputs)
    result_rot_x = dft_qnn.apply(parameters, rot_mock_coeff_inputs_x)
    result_rot_y = dft_qnn.apply(parameters, rot_mock_coeff_inputs_y)
    result_rot_z = dft_qnn.apply(parameters, rot_mock_coeff_inputs_z)
    result_rot_z_rot_x = dft_qnn.apply(parameters, rot_mock_coeff_inputs_z_inputs_x)

    assert np.allclose(result_rot_x, result, atol=1e-6)
    assert np.allclose(result_rot_y, result, atol=1e-6)
    assert np.allclose(result_rot_z, result, atol=1e-6)
    assert np.allclose(result_rot_z_rot_x, result, atol=1e-6)
