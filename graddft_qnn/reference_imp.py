import json
import logging
import pathlib
from datetime import datetime

import grad_dft as gd
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pennylane as qml
import yaml
from jax.random import PRNGKey
from optax import adam, apply_updates
from pyscf import dft, gto
from tqdm import tqdm

from datasets import DatasetDict
from graddft_qnn import helper, unitary_rep
from graddft_qnn.cube_dataset.h2_multibond import H2MultibondDataset
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.evaluate.metric_name import MetricName
from graddft_qnn.io.ansatz_io import AnsatzIO
from graddft_qnn.naive_dft_qnn import NaiveDFTQNN
from graddft_qnn.qnn_functional import QNNFunctional
from graddft_qnn.unitary_rep import O_h, is_group

logging.getLogger().setLevel(logging.INFO)
np.random.seed(42)
key = PRNGKey(42)


@jax.jit
def coefficient_inputs(molecule: gd.Molecule, *_, **__):
    rho = molecule.density()
    kinetic = molecule.kinetic_density()
    return jnp.sum(rho, axis=1) + jnp.sum(kinetic, axis=1)


@jax.jit
def energy_densities(molecule: gd.Molecule, clip_cte: float = 1e-30, *_, **__):
    rho = jnp.clip(molecule.density(), a_min=clip_cte)
    lda_e = (
        -3
        / 2
        * (3 / (4 * jnp.pi)) ** (1 / 3)
        * (rho ** (4 / 3)).sum(axis=1, keepdims=True)
    )
    return lda_e


out_features = 1

logging.getLogger().setLevel(logging.INFO)
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
    if "naive" not in group[0].lower():
        group_str_rep = "]_[".join(group)[:230]
        group_matrix_reps = [getattr(O_h, gr)(size, False) for gr in group]
        if (check_group) and (not is_group(group_matrix_reps, group)):
            raise ValueError("Not forming a group")
    xc_functional_name = data["XC_FUNCTIONAL"]
    dev = qml.device("default.qubit", wires=num_qubits)

# define the QNN
if "naive" not in group[0].lower():
    filename = f"ansatz_{num_qubits}_{group_str_rep}_qubits"
    if pathlib.Path(f"{filename}.pkl").exists():
        gates_gen = AnsatzIO.read_from_file(filename)
        logging.info(f"Loaded ansatz generator from {filename}")
    else:
        gates_gen = unitary_rep.gate_design(
            len(dev.wires), [getattr(O_h, gr)(size, True) for gr in group]
        )
        AnsatzIO.write_to_file(filename, gates_gen)
    gates_gen = gates_gen[: 2**num_qubits]
    if isinstance(num_gates, int):
        gates_indices = sorted(np.random.choice(len(gates_gen), num_gates))
    dft_qnn = DFTQNN(dev, gates_gen, gates_indices)
else:
    dft_qnn = NaiveDFTQNN(dev, num_gates)

nf = QNNFunctional(dft_qnn, energy_densities, coefficient_inputs)


key = PRNGKey(42)
coeff_input = jnp.empty((2 ** len(dev.wires),))
params = dft_qnn.init(key, coeff_input)

if pathlib.Path("datasets/h2_dataset").exists():
    dataset = DatasetDict.load_from_disk(pathlib.Path("datasets/h2_dataset"))
else:
    dataset = H2MultibondDataset.get_dataset()
    dataset.save_to_disk("datasets/h2_dataset")

tx = adam(learning_rate=learning_rate, b1=momentum)
opt_state = tx.init(params)

predictor = gd.non_scf_predictor(nf)

train_losses = []
test_losses = []
train_ds = dataset["train"]
for epoch in tqdm(range(n_epochs)):
    train_ds = train_ds.shuffle(seed=42)
    aggregated_train_loss = 0
    for batch in train_ds:
        atom_coords = list(zip(batch["symbols"], batch["coordinates"]))
        mol = gto.M(atom=atom_coords, basis="def2-tzvp")
        mean_field = dft.UKS(mol)
        mean_field.kernel()
        molecule = gd.molecule_from_pyscf(mean_field)
        (cost_value, predicted_energy), grads = gd.simple_energy_loss(
            params, predictor, molecule, batch["groundtruth"]
        )
        E = nf.energy(params, molecule)
        # print("QNN functional energy parameters is", E)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = apply_updates(params, updates)
        print(params)
        aggregated_train_loss += cost_value

    train_loss = np.sqrt(aggregated_train_loss / len(train_ds))
    logging.info(f"RMS train loss: {train_loss}")
    train_losses.append(train_loss)


aggregated_cost = 0
for batch in tqdm(dataset["test"], desc="Evaluate"):
    cost_value = helper.training.eval_step(params, predictor, batch)
    aggregated_cost += cost_value
test_loss = np.sqrt(aggregated_cost / len(dataset["test"]))
logging.info(f"Test loss {test_loss}")

# report
now = datetime.now()
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
report = {
    MetricName.DATE: date_time,
    MetricName.N_QUBITS: num_qubits,
    MetricName.TEST_LOSS: test_loss,
    MetricName.N_GATES: num_gates,
    MetricName.N_MEASUREMENTS: full_measurements,
    MetricName.GROUP_MEMBER: group,
    MetricName.EPOCHS: n_epochs,
    MetricName.TRAIN_LOSSES: train_losses,
    MetricName.TEST_LOSSES: test_losses,
    MetricName.LEARNING_RATE: learning_rate,
    MetricName.BATCH_SIZE: batch_size,
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
