"""
This will
1. read the config specified in config.yaml
2. load the checkpoint associated with that config
3. do annotation
"""

import logging
import pathlib

import grad_dft as gd
import jax
import numpy as np
import pennylane as qml
import tqdm
from optax import adam
from pyscf import dft, gto

from graddft_qnn import helper
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.io.ansatz_io import AnsatzIO
from graddft_qnn.naive_dft_qnn import NaiveDFTQNN
from graddft_qnn.qnn_functional import QNNFunctional
from graddft_qnn.unitary_rep import O_h, is_group

logging.getLogger().setLevel(logging.INFO)
np.random.seed(42)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    (
        num_qubits,
        size,
        n_epochs,
        learning_rate,
        momentum,
        eval_per_x_epoch,
        batch_size,
        group,
        xc_functional_name,
        check_group,
        num_gates,
    ) = helper.initialization.load_config("config.yaml")
    if "naive" not in group[0].lower():
        group_str_rep = "]_[".join(group)[:230]
        group_matrix_reps = [getattr(O_h, gr)(size, False) for gr in group]
        if (check_group) and (not is_group(group_matrix_reps, group)):
            raise ValueError("Not forming a group")
    dev = qml.device("default.qubit", wires=num_qubits)

    # define the QNN
    filename = f"ansatz_{num_qubits}_{group_str_rep}_qubits"
    if "naive" not in group[0].lower():
        if pathlib.Path(filename).exists():
            gates_gen = AnsatzIO.read_from_file(filename)
            logging.info(f"Loaded ansatz generator from {filename}")
        else:
            gates_gen = DFTQNN.gate_design(
                len(dev.wires), [getattr(O_h, gr)(size, True) for gr in group]
            )
            AnsatzIO.write_to_file(filename, gates_gen)
        gates_gen = gates_gen[: 2**num_qubits]
        if isinstance(num_gates, int):
            gates_indices = sorted(np.random.choice(len(gates_gen), num_gates))
        dft_qnn = DFTQNN(dev, gates_gen, gates_indices)
    else:
        z_measurements = NaiveDFTQNN.generate_Z_measurements(len(dev.wires))
        dft_qnn = NaiveDFTQNN(dev, z_measurements, num_gates)

    # define the functional
    qnnf = QNNFunctional(
        coefficients=dft_qnn,
        energy_densities=helper.initialization.energy_densities,
        coefficient_inputs=helper.initialization.coefficient_inputs,
    )

    checkpoint_path = (
        pathlib.Path().resolve() / pathlib.Path(filename).stem / "checkpoint_1"
    )
    tx = adam(learning_rate=learning_rate, b1=momentum)
    state = qnnf.load_checkpoint(tx, ckpt_dir=str(checkpoint_path))

    parameters = state.params
    tx = state.tx
    opt_state = tx.init(parameters)
    epoch = state.step

    # Evaluate the model
    predictor = gd.non_scf_predictor(qnnf)
    distances = np.arange(0.3, 3.0, 0.5)
    predicts = []
    for distance in tqdm.tqdm(distances, desc="Calculating Binding Energy"):
        # Create molecule with the specified distance
        mol = gto.M(
            atom=[["H", (0, 0, 0)], ["H", (0, 0, distance)]],
            basis="def2-tzvp",
            unit="Angstrom",
        )
        mean_field = dft.UKS(mol)
        mean_field.xc = "wB97M-V"
        mean_field.nlc = "VV10"

        ground_truth_energy = mean_field.kernel()
        molecule = gd.molecule_from_pyscf(mean_field)

        (cost_value, predicted_energy), _ = gd.simple_energy_loss(
            parameters, predictor, molecule, ground_truth_energy
        )
        predicts.append(predicted_energy)
    helper.visualization.plot_list(distances, predicts)
