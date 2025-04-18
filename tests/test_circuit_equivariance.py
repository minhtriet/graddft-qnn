import pathlib

import pennylane as qml
import pennylane.numpy as np
from jax.random import PRNGKey

from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.io.ansatz_io import AnsatzIO
from graddft_qnn.unitary_rep import O_h


def test_a_training_step():
    num_wires = 3
    _setup_device = qml.device("default.qubit", num_wires)
    gates_indices = list(np.random.choice(2**num_wires, num_wires, replace=False))
    filename = pathlib.Path("tests") / "ansatz_3_qubits_180_x.pkl"
    gates_gen = AnsatzIO.read_from_file(str(filename))
    measurement_expvals = gates_gen
    mock_coeff_inputs = np.random.rand(2**num_wires)
    dft_qnn = DFTQNN(_setup_device, gates_gen, measurement_expvals, gates_indices)

    rot_mock_coeff_inputs_x = (
        O_h._180_deg_x_rot(int(np.cbrt(2**num_wires))) @ mock_coeff_inputs
    )
    # rot_mock_coeff_inputs_y = (
    #     O_h._180_deg_y_rot(int(np.cbrt(2**num_wires))) @ mock_coeff_inputs
    # )
    # rot_mock_coeff_inputs_z = (
    #     O_h._180_deg_z_rot(int(np.cbrt(2**num_wires))) @ mock_coeff_inputs
    # )

    key = PRNGKey(42)

    parameters = dft_qnn.init(key, mock_coeff_inputs)

    result = dft_qnn.apply(parameters, mock_coeff_inputs)
    result_rot_x = dft_qnn.apply(parameters, rot_mock_coeff_inputs_x)
    # result_rot_y = dft_qnn.apply(parameters, rot_mock_coeff_inputs_y)
    # result_rot_z = dft_qnn.apply(parameters, rot_mock_coeff_inputs_z)

    assert np.allclose(result_rot_x, result)
    # assert np.allclose(result_rot_y, result, atol=1e-6)
    # assert np.allclose(result_rot_z, result, atol=1e-6)
    # assert np.allclose(result_rot_z_rot_x, result, atol=1e-6)
