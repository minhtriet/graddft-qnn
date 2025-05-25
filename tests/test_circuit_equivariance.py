import pathlib

import grad_dft as gd
import jax.numpy as jnp
import pennylane as qml
import pennylane.numpy as np
from grad_dft import abs_clip
from jax.lax import Precision
from jax.random import PRNGKey, normal
from optax import adam

from datasets import DatasetDict
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.helper import initialization, training
from graddft_qnn.io.ansatz_io import AnsatzIO
from graddft_qnn.qnn_functional import QNNFunctional
from graddft_qnn.unitary_rep import O_h


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


def _prepare_dataset():
    dataset = DatasetDict.load_from_disk(pathlib.Path("datasets/h2_dataset"))
    return dataset["train"]


def test_a_training_step_6qb_d4():
    """
    After rotate a QNN input feature, the loss function
    calculated based on E_ks should be the same as rotate the output of QNN
    """
    num_wires = 6
    num_gates = 8
    np.random.seed(17)
    _setup_device = qml.device("default.qubit", num_wires)
    filename = pathlib.Path("tests") / "ansatz_6_d4"
    gates_gen = AnsatzIO.read_from_file(str(filename))
    gates_indices = list(np.random.choice(len(gates_gen), num_gates, replace=False))
    mock_params = jnp.empty((2**num_wires,))
    dataset = _prepare_dataset()
    key = PRNGKey(42)
    _270_x_y_eq_z = O_h._270_deg_x_rot(int(np.cbrt(2**num_wires))) @ O_h.y_eq_z_rot(
        int(np.cbrt(2**num_wires))
    )

    # This is a QNN that always rotate the result of the QNN by _270_x_y_eq_z
    dft_qnn_nothing = DFTQNN(
        _setup_device, gates_gen, gates_indices, _270_x_y_eq_z, rotate_feature=False
    )
    qnnf_nothing = QNNFunctional(
        coefficients=dft_qnn_nothing,
        energy_densities=initialization.energy_densities,
        coefficient_inputs=initialization.coefficient_inputs,
    )

    # This is a QNN that always rotate the result of the QNN by _270_x_y_eq_z
    dft_qnn_rot_result = DFTQNN(
        _setup_device, gates_gen, gates_indices, _270_x_y_eq_z, rotate_feature=False
    )
    qnnf_rot_result = QNNFunctional(
        coefficients=dft_qnn_rot_result,
        energy_densities=initialization.energy_densities,
        coefficient_inputs=initialization.coefficient_inputs,
    )

    # This is a QNN that always rotate input feature of the QNN by _270_x_y_eq_z
    dft_qnn_rot_input = DFTQNN(
        _setup_device, gates_gen, gates_indices, _270_x_y_eq_z, rotate_feature=True
    )
    qnnf_rot_input = QNNFunctional(
        coefficients=dft_qnn_rot_input,
        energy_densities=initialization.energy_densities,
        coefficient_inputs=initialization.coefficient_inputs,
    )

    parameters = dft_qnn_rot_result.init(key, mock_params)
    tx = adam(learning_rate=0.1, b1=0.9)
    opt_state = tx.init(parameters)

    predictor = gd.non_scf_predictor(qnnf_nothing)
    _, _, avg_cost = training.train_step(
        parameters, predictor, dataset[:1], opt_state, tx
    )

    predictor_rot_input = gd.non_scf_predictor(qnnf_rot_input)
    _, _, avg_cost_2 = training.train_step(
        parameters, predictor_rot_input, dataset[:1], opt_state, tx
    )

    predictor_rot_result = gd.non_scf_predictor(qnnf_rot_result)
    _, _, avg_cost_3 = training.train_step(
        parameters, predictor_rot_result, dataset[:1], opt_state, tx
    )

    # Assert that loss(rotate input -> QNN) == loss(QNN -> rotate output)
    # which is the definition of equivariance
    assert np.isclose(avg_cost_2, avg_cost_3)

    # this should be close because lost is invariant
    assert np.isclose(avg_cost, avg_cost_3)


def test_180_x_rot_matrix():
    a = O_h._180_deg_x_rot_sparse(4, False).todense()
    b = O_h._180_deg_x_rot(4, False)
    assert np.allclose(a, b)


def test_a_training_step_6qb_d4_2():
    num_wires = 6
    np.random.seed(17)
    _setup_device = qml.device("default.qubit", num_wires)
    gates_indices = list(np.random.choice(2**num_wires, num_wires, replace=False))
    filename = pathlib.Path("tests") / "ansatz_6_d4"
    gates_gen = AnsatzIO.read_from_file(str(filename))
    mock_coeff_inputs = np.random.rand(2**num_wires)
    dft_qnn = DFTQNN(_setup_device, gates_gen, gates_indices)
    _180_x_rot = O_h._180_deg_x_rot(int(np.cbrt(2**num_wires)))
    _180_y_rot = O_h._180_deg_y_rot(int(np.cbrt(2**num_wires)))
    _180_z_rot = O_h._180_deg_z_rot(int(np.cbrt(2**num_wires)))
    _270_x_y_eq_z = O_h._270_deg_x_rot(int(np.cbrt(2**num_wires))) @ O_h.y_eq_z_rot(
        int(np.cbrt(2**num_wires))
    )
    rot_mock_coeff_inputs_x = _180_x_rot @ mock_coeff_inputs
    rot_mock_coeff_inputs_y = _180_y_rot @ mock_coeff_inputs
    rot_mock_coeff_inputs_z = _180_z_rot @ mock_coeff_inputs
    rot_mock_coeff_inputs_x_y_eq_z = _270_x_y_eq_z @ mock_coeff_inputs
    key = PRNGKey(42)

    parameters = dft_qnn.init(key, mock_coeff_inputs)

    result = dft_qnn.apply(parameters, mock_coeff_inputs)
    result_rot_x = dft_qnn.apply(parameters, rot_mock_coeff_inputs_x)
    result_rot_y = dft_qnn.apply(parameters, rot_mock_coeff_inputs_y)
    result_rot_z = dft_qnn.apply(parameters, rot_mock_coeff_inputs_z)
    result_rot_mock_coeff_inputs_x_y_eq_z = dft_qnn.apply(
        parameters, rot_mock_coeff_inputs_x_y_eq_z
    )

    assert np.allclose(result_rot_x, _180_x_rot @ result)
    assert np.allclose(result_rot_y, _180_y_rot @ result)
    assert np.allclose(result_rot_z, _180_z_rot @ result)
    assert np.allclose(result_rot_mock_coeff_inputs_x_y_eq_z, _270_x_y_eq_z @ result)

    densities = normal(key, shape=(2**num_wires, 1))

    result = result[:, jnp.newaxis]
    result_rot_x = result_rot_x[:, jnp.newaxis]

    xc_energy_density = jnp.einsum("rf,rf->r", result, densities)
    xc_energy_density = abs_clip(xc_energy_density, 1e-30)

    xc_energy_density_rot_x = jnp.einsum(
        "rf,rf->r", result_rot_x, _180_x_rot @ densities
    )
    xc_energy_density_rot_x = abs_clip(xc_energy_density_rot_x, 1e-30)
    grid_weights = np.full((2**num_wires,), 1 / (2**num_wires))
    assert np.isclose(
        _integrate(xc_energy_density, grid_weights),
        _integrate(xc_energy_density_rot_x, grid_weights),
    )


def test_270_x_rot_sparse_matrix():
    num_wire = 6
    dev = qml.device("default.qubit", wires=num_wire)

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
