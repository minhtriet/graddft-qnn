import pathlib

import jax.numpy as jnp
import pennylane as qml
import pennylane.numpy as np
import pytest
from grad_dft import abs_clip
from jax.lax import Precision
from jax.random import PRNGKey, normal

from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.io.ansatz_io import AnsatzIO
from graddft_qnn.unitary_rep import O_h


def test_a_training_step():
    num_wires = 3
    np.random.seed(17)
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

    key = PRNGKey(42)

    parameters = dft_qnn.init(key, mock_coeff_inputs)

    result = dft_qnn.apply(parameters, mock_coeff_inputs)
    result_rot_x = dft_qnn.apply(parameters, rot_mock_coeff_inputs_x)
    assert np.allclose(result_rot_x, result)


def _integrate(energy_density, gridweights, clip_cte=1e-30):
    """
    copy verbatim from grad_dft.functional.Functional._integrate, this could
    have been a static function
    """
    return jnp.einsum(
        "r,r->",
        abs_clip(gridweights, clip_cte),
        abs_clip(energy_density, clip_cte),
        precision=Precision.HIGHEST,
    )


@pytest.mark.skip(reason="training work, but this has a strange JAX leak error")
def test_a_training_step_6qb_d4():
    num_wires = 6
    np.random.seed(17)
    _setup_device = qml.device("default.qubit", num_wires)
    gates_indices = list(np.random.choice(2**num_wires, num_wires, replace=False))
    filename = pathlib.Path("tests") / "ansatz_6_d4.pkl"
    gates_gen = AnsatzIO.read_from_file(str(filename))
    measurement_expvals = gates_gen
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
    rot_mock_coeff_inputs_x_y_eq_z = (
        O_h._270_deg_x_rot(int(np.cbrt(2**num_wires)))
        @ O_h.y_eq_z_rot(int(np.cbrt(2**num_wires)))
        @ mock_coeff_inputs
    )
    key = PRNGKey(42)

    parameters = dft_qnn.init(key, mock_coeff_inputs)

    result = dft_qnn.apply(parameters, mock_coeff_inputs)
    result_rot_x = dft_qnn.apply(parameters, rot_mock_coeff_inputs_x)
    result_rot_y = dft_qnn.apply(parameters, rot_mock_coeff_inputs_y)
    result_rot_z = dft_qnn.apply(parameters, rot_mock_coeff_inputs_z)
    result_rot_mock_coeff_inputs_x_y_eq_z = dft_qnn.apply(
        parameters, rot_mock_coeff_inputs_x_y_eq_z
    )

    assert np.allclose(result_rot_x, result)
    assert np.allclose(result_rot_y, result)
    assert np.allclose(result_rot_z, result)
    assert np.allclose(result_rot_mock_coeff_inputs_x_y_eq_z, result)

    densities = normal(key, shape=(2**num_wires, 1))

    result = result[:, jnp.newaxis]
    result_rot_x = result_rot_x[:, jnp.newaxis]

    xc_energy_density = jnp.einsum("rf,rf->r", result, densities)
    xc_energy_density = abs_clip(xc_energy_density, 1e-30)

    xc_energy_density_rot_x = jnp.einsum("rf,rf->r", result_rot_x, densities)
    xc_energy_density_rot_x = abs_clip(xc_energy_density_rot_x, 1e-30)

    grid_weights = normal(key, shape=(2**num_wires,))
    assert np.isclose(
        _integrate(xc_energy_density, grid_weights),
        _integrate(xc_energy_density_rot_x, grid_weights),
    )


def test_270_x_rot_sparse_matrix():
    num_wire = 6
    dev = qml.device("lightning.qubit", wires=num_wire)

    @qml.qnode(dev)
    def six_qubit_circuit_dense(params):
        qml.AmplitudeEmbedding(params, wires=range(6), normalize=True)
        qml.X(0)
        qml.Y(1)
        qml.RZ(1.23, 2)
        return qml.expval(O_h._270_deg_x_rot(4, pauli_word=True))

    @qml.qnode(dev)
    def six_qubit_circuit_sparse(params):
        qml.AmplitudeEmbedding(params, wires=range(6), normalize=True)
        qml.X(0)
        qml.Y(1)
        qml.RZ(1.23, 2)
        return qml.expval(O_h._270_deg_x_rot_sparse(4, pauli_word=True))

    np.random.seed(14)
    mock_coeff_inputs = np.random.rand(2**num_wire)
    dense_result = six_qubit_circuit_dense(mock_coeff_inputs)
    sparse_result = six_qubit_circuit_sparse(mock_coeff_inputs)
    assert np.allclose(dense_result, sparse_result)


def test_180_x_rot_matrix():
    a = O_h._180_deg_x_rot_sparse(4, False).todense()
    b = O_h._180_deg_x_rot(4, False)
    assert np.allclose(a, b)
