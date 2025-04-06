import json
import logging
import pathlib
from datetime import datetime

import grad_dft as gd
import jax
import numpy as np
import pandas as pd
import pennylane as qml
import tqdm
import yaml
from evaluate.metric_name import MetricName
from jax import numpy as jnp
from jax.random import PRNGKey
from jaxtyping import PyTree
from optax import adam, apply_updates
from pyscf import dft, gto
from unitary_rep import is_group

from datasets import DatasetDict
from graddft_qnn import custom_gates
from graddft_qnn.cube_dataset.cube_dataset_hf import CubeDataset
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.io.ansatz_io import AnsatzIO
from graddft_qnn.qnn_functional import QNNFunctional
from graddft_qnn.unitary_rep import O_h

logging.getLogger().setLevel(logging.INFO)
np.random.seed(42)
key = PRNGKey(42)


@jax.jit
def coefficient_inputs(molecule: gd.Molecule, *_, **__):
    rho = molecule.density()
    # change: total spin, also why it is not +/-0.5?
    return jnp.sum(rho, 1)


@jax.jit
def energy_densities(molecule: gd.Molecule, clip_cte: float = 1e-30, *_, **__):
    r"""Auxiliary function to generate the features of LSDA."""
    # Molecule can compute the density matrix.
    rho = jnp.clip(molecule.density(), a_min=clip_cte)
    # Now we can implement the LDA energy density equation in the paper.
    lda_e = (
        -3
        / 2
        * (3 / (4 * jnp.pi)) ** (1 / 3)
        * (rho ** (4 / 3)).sum(axis=1, keepdims=True)
    )
    # For simplicity we do not include the exchange polarization correction
    # check function exchange_polarization_correction in functional.py
    # The output of features must be an Array of dimension n_grid x n_features.
    return lda_e


def simple_energy_loss(
    params: PyTree,
    compute_energy,  #:  Callable,
    atoms,  #: #Union[Molecule, Solid],
    truth_energy,  #: #Float,
):
    """
    Computes the loss for a single molecule

    Parameters
    ----------
    params: PyTree
        functional parameters (weights)
    compute_energy: Callable.
        any non SCF or SCF method in evaluate.py
    atoms: Union[Molecule, Solid]
        The collcection of atoms.
    truth_energy: Float
        The energy value we are training against
    """
    atoms_out = compute_energy(params, atoms)
    E_predict = atoms_out.energy
    diff = E_predict - truth_energy
    return diff**2, E_predict


if __name__ == "__main__":
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
        assert (
            isinstance(num_gates, int) or num_gates == "full"
        ), f"N_GATES must be integer or 'full', got {num_gates}"
        full_measurements = data["FULL_MEASUREMENTS"]
        group = data["GROUP"]
        group_str_rep = "]_[".join(group)[:230]
        group_matrix_reps = [getattr(O_h, gr)(size, False) for gr in group]
        if not is_group(group_matrix_reps, group):
            raise ValueError("Not forming a group")
        dev = qml.device("default.qubit", wires=num_qubits)

    # config model params
    jax.config.update("jax_enable_x64", True)

    # define the QNN
    filename = f"ansatz_{num_qubits}_{group_str_rep}_qubits.txt"
    if pathlib.Path(filename).exists():
        gates_gen = AnsatzIO.read_from_file(filename)
        logging.info(f"Loaded ansatz generator from {filename}")
    else:
        gates_gen = DFTQNN.gate_design(
            len(dev.wires), [getattr(O_h, gr)(size, True) for gr in group]
        )
        AnsatzIO.write_to_file(filename, gates_gen)
    gates_gen = gates_gen[: 2**num_qubits]
    if full_measurements:
        measurement_expvals = [
            custom_gates.generate_operators(measurement) for measurement in gates_gen
        ]
    else:
        measurement_expvals = [
            custom_gates.generate_operators(measurement)
            for measurement in gates_gen[:1]
        ]
    if isinstance(num_gates, int):
        gates_indices = sorted(np.random.choice(len(gates_gen), num_gates))
    dft_qnn = DFTQNN(dev, gates_gen, measurement_expvals, gates_indices)

    # get a sample batch for initialization
    coeff_input = jnp.zeros((2 ** len(dev.wires),))
    logging.info("Initializing the params")
    parameters = dft_qnn.init(key, coeff_input)
    logging.info("Finished initializing the params")

    # define the functional
    nf = QNNFunctional(
        coefficients=dft_qnn,
        energy_densities=energy_densities,
        coefficient_inputs=coefficient_inputs,
    )
    tx = adam(learning_rate=learning_rate, b1=momentum)
    opt_state = tx.init(parameters)

    predictor = gd.non_scf_predictor(nf)
    # start training
    if pathlib.Path("datasets/hf_dataset").exists():
        dataset = DatasetDict.load_from_disk(pathlib.Path("datasets/hf_dataset"))
    else:
        dataset = CubeDataset.get_dataset()
        dataset.save_to_disk("datasets/hf_dataset")

    # train
    train_losses = []
    train_ds = dataset["train"]
    for epoch in range(n_epochs):
        train_ds = train_ds.shuffle(seed=42)
        aggregated_train_loss = 0
        for batch in tqdm.tqdm(train_ds, desc=f"Epoch {epoch + 1}"):
            atom_coords = list(zip(batch["symbols"], batch["coordinates"]))
            mol = gto.M(atom=atom_coords, basis="def2-tzvp")
            mean_field = dft.UKS(mol)
            mean_field.kernel()
            molecule = gd.molecule_from_pyscf(mean_field)
            (cost_value, predicted_energy), grads = gd.simple_energy_loss(
                parameters, predictor, molecule, batch["groundtruth"]
            )
            aggregated_train_loss += cost_value
            # print(
            #     "Predicted energy:",
            #     predicted_energy,
            #     "Cost value:",
            #     cost_value,
            #     "Grad: ",
            #     (jnp.max(grads["params"]["theta"]), jnp.min(grads["params"]["theta"])),
            # )
            updates, opt_state = tx.update(grads, opt_state, parameters)
            parameters = apply_updates(parameters, updates)
        train_loss = np.sqrt(aggregated_train_loss / len(train_ds))
        logging.info(f"RMS loss: {train_loss}")
        train_losses.append(train_loss)
    logging.info("Start evaluating")
    # test
    aggregated_cost = 0
    for batch in tqdm.tqdm(dataset["test"], desc="Evaluate"):
        atom_coords = list(zip(batch["symbols"], batch["coordinates"]))
        mol = gto.M(atom=atom_coords, basis="def2-tzvp")
        mean_field = dft.UKS(mol)
        mean_field.kernel()
        molecule = gd.molecule_from_pyscf(mean_field)
        (cost_value, predicted_energy), grads = gd.simple_energy_loss(
            parameters, predictor, molecule, batch["groundtruth"]
        )
        aggregated_cost += cost_value
    test_loss = np.sqrt(aggregated_cost / len(dataset["test"]))
    logging.info(f"Test loss {test_loss}")
    # report
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    report = {
        MetricName.DATE: date_time,
        MetricName.N_QUBITS: num_qubits,
        MetricName.TEST_LOSSES: test_loss,
        MetricName.N_GATES: num_gates,
        MetricName.N_MEASUREMENTS: full_measurements,
        MetricName.GROUP_MEMBER: group,
        MetricName.EPOCHS: n_epochs,
        MetricName.TRAIN_LOSSES: train_losses,
        MetricName.LEARNING_RATE: learning_rate
    }
    if pathlib.Path("report.json").exists():
        with open("report.json") as f:
            try:
                history_report = json.load(f)
            except json.decoder.JSONDecodeError:
                history_report = []
    else:
        history_report = []
    history_report.append(report)
    with open("report.json", "w") as f:
        json.dump(history_report, f)
    pd.DataFrame(history_report).to_excel("report.xlsx")

"""
1. do 2 3 molecules, compare to classical
2. increase finess of grids
3. log the xc_energy + total_energy
4. regularization (params 0 - 2pi)
# todo calculate the number of symmetric equivalent pixel rather than all pixels
----
25th
Make sure Jax has same num of params, downsized grid
Metric of success
- Testing on unseen data
- Add / remove symetry for the ansatz
    - gradients steeper / smaller number of interations when adding more group
"""
