"""
This will
1. read the config specified in config.yaml
2. load the checkpoint associated with that config
3. do annotation
"""
import json
import logging
import pathlib

import grad_dft as gd
import jax
import numpy as np
import pennylane as qml
import tqdm
import yaml
from helper.visualization import DISTANCES, h2_dist_energy
from optax import adamw
from pyscf import dft, gto

from graddft_qnn import helper
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.io.ansatz_io import AnsatzIO
from graddft_qnn.naive_dft_qnn import NaiveDFTQNN
from graddft_qnn.qnn_functional import QNNFunctional
from graddft_qnn.unitary_rep import O_h, is_group

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    with open("config.yaml") as file:
        data = yaml.safe_load(file)
        if "QBITS" not in data:
            raise KeyError("YAML file must contain 'QBITS' key")
        num_qubits = data["QBITS"]
        size = np.cbrt(2**num_qubits)
        assert size.is_integer()
        size = int(size)
        n_epochs = data["TRAINING"]["N_EPOCHS"]
        learning_rate = data["TRAINING"]["LEARNING_RATE"]
        momentum = data["TRAINING"]["MOMENTUM"]
        num_gates = data["N_GATES"]
        eval_per_x_epoch = data["TRAINING"]["EVAL_PER_X_EPOCH"]
        batch_size = data["TRAINING"]["BATCH_SIZE"]
        check_group = data["CHECK_GROUP"]
        assert (
            isinstance(num_gates, int) or num_gates == "full"
        ), f"N_GATES must be integer or 'full', got {num_gates}"
        full_measurements = "prob"
        group: list = data["GROUP"]
        group_str_rep = "]_[".join(group)[:230]
        if "naive" not in group[0].lower():
            group_matrix_reps = [getattr(O_h, gr)(size, False) for gr in group]
            if (check_group) and (not is_group(group_matrix_reps, group)):
                raise ValueError("Not forming a group")
        xc_functional_name = data["XC_FUNCTIONAL"]
        dev = qml.device("default.qubit", wires=num_qubits)

    with open(helper.visualization.CLASSICAL_WITH_DOWN_FILENAME) as f:
        classical = json.load(f)

    # define the QNN
    filename = f"ansatz_{num_qubits}_{group_str_rep}_qubits"
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

    # always load naive and the specified one in config.yaml
    dft_qnn_naive = NaiveDFTQNN(dev, num_gates)

    # define the functional
    qnnf = QNNFunctional(
        coefficients=dft_qnn,
        energy_densities=helper.initialization.energy_densities,
        coefficient_inputs=helper.initialization.coefficient_inputs,
    )
    checkpoint_path = (
        pathlib.Path().resolve() / pathlib.Path(filename).stem / f"checkpoint_{n_epochs}"
    )
    tx = adamw(learning_rate=learning_rate, b1=momentum)
    state = qnnf.load_checkpoint(tx, ckpt_dir=str(checkpoint_path))

    qnnf_naive = QNNFunctional(
        coefficients=dft_qnn_naive,
        energy_densities=helper.initialization.energy_densities,
        coefficient_inputs=helper.initialization.coefficient_inputs,
    )
    checkpoint_path = (
            pathlib.Path().resolve() / pathlib.Path(filename).stem / f"checkpoint_{n_epochs}"
    )
    tx_naive = adamw(learning_rate=learning_rate, b1=momentum)
    state_naive = qnnf_naive.load_checkpoint(tx, ckpt_dir=str(checkpoint_path))

    parameters = state.params
    parameters_naive = state_naive.params
    tx = state.tx
    epoch = state.step

    # Evaluate the model
    predictor = gd.non_scf_predictor(qnnf)
    predictor_naive = gd.non_scf_predictor(qnnf_naive)

    predicts = dict()
    predicts_naive = dict()
    for distance in tqdm.tqdm(DISTANCES, desc="Calculating Binding Energy"):
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

        (_, predicted_energy), _ = gd.simple_energy_loss(
            parameters, predictor, molecule, ground_truth_energy
        )
        (_, predicted_naive_energy), _ = gd.simple_energy_loss(
            parameters_naive, predictor_naive, molecule, ground_truth_energy
        )
        predicts[distance] = predicted_energy
        predicts_naive[distance] = predicted_naive_energy

    helper.visualization.plot_list(
        [h2_dist_energy(), predicts, classical],
        ["Invariant quantum", "Classical"],
    )
