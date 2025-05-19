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
from evaluate.metric_name import MetricName
from jax import numpy as jnp
from jax.random import PRNGKey
from jaxtyping import PyTree
from optax import adam

from datasets import DatasetDict
from graddft_qnn import helper
from graddft_qnn.cube_dataset.h2_multibond import H2MultibondDataset
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.io.ansatz_io import AnsatzIO
from graddft_qnn.naive_dft_qnn import NaiveDFTQNN
from graddft_qnn.qnn_functional import QNNFunctional
from graddft_qnn.unitary_rep import O_h, is_group

logging.getLogger().setLevel(logging.INFO)
np.random.seed(42)
key = PRNGKey(42)


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
        if pathlib.Path(f"{filename}.pkl").exists():
            gates_gen = AnsatzIO.read_from_file(filename)
            assert (
                len(gates_gen) >= num_gates
            ), "Pickled file has less gates than needed, please delete and regenerate"
            logging.info(f"Loaded ansatz generator from {filename}")
        else:
            gates_gen = DFTQNN.gate_design(
                len(dev.wires),
                [getattr(O_h, gr)(size, True) for gr in group],
                num_generator=num_gates,
            )
            AnsatzIO.write_to_file(filename, gates_gen)
        if isinstance(num_gates, int):
            gates_indices = sorted(np.random.choice(len(gates_gen), num_gates))
        dft_qnn = DFTQNN(dev, gates_gen, gates_indices)
    else:
        z_measurements = NaiveDFTQNN.generate_Z_measurements(len(dev.wires))
        dft_qnn = NaiveDFTQNN(dev, z_measurements, num_gates)

    # get a sample batch for initialization
    coeff_input = jnp.zeros((2 ** len(dev.wires),))
    logging.info("Initializing the params")
    parameters = dft_qnn.init(key, coeff_input)
    logging.info("Finished initializing the params")

    # resolve energy density according to user input
    e_density = helper.initialization.resolve_energy_density(xc_functional_name)

    # define the functional
    qnnf = QNNFunctional(
        coefficients=dft_qnn,
        energy_densities=helper.initialization.energy_densities,
        coefficient_inputs=helper.initialization.coefficient_inputs,
    )
    tx = adam(learning_rate=learning_rate, b1=momentum)
    opt_state = tx.init(parameters)

    predictor = gd.non_scf_predictor(qnnf)
    # start training
    if pathlib.Path("datasets/h2_dataset").exists():
        dataset = DatasetDict.load_from_disk(pathlib.Path("datasets/h2_dataset"))
    else:
        dataset = H2MultibondDataset.get_dataset()
        dataset.save_to_disk("datasets/h2_dataset")

    # train
    train_losses = []
    test_losses = []
    train_ds = dataset["train"]
    for epoch in range(n_epochs):
        train_ds = train_ds.shuffle(seed=42)
        aggregated_train_loss = 0

        for i in tqdm.tqdm(
            range(0, len(train_ds), batch_size), desc=f"Epoch {epoch + 1}"
        ):
            batch = train_ds[i : i + batch_size]
            parameters, opt_state, cost_value = helper.training.train_step(
                parameters, predictor, batch, opt_state, tx
            )
            aggregated_train_loss += cost_value

        train_loss = np.sqrt(aggregated_train_loss / len(train_ds))
        logging.info(f"RMS train loss: {train_loss}")
        train_losses.append(train_loss)

        if (epoch + 1) % eval_per_x_epoch == 0:
            aggregated_cost = 0
            for batch in tqdm.tqdm(
                dataset["test"], desc=f"Evaluate per {eval_per_x_epoch} epoch"
            ):
                cost_value = helper.training.eval_step(parameters, predictor, batch)
                aggregated_cost += cost_value
            test_loss = np.sqrt(aggregated_cost / len(dataset["test"]))
            test_losses.append({epoch: test_loss})
            logging.info(f"Test loss: {test_loss}")

    logging.info("Start evaluating")
    # test
    aggregated_cost = 0
    for batch in tqdm.tqdm(dataset["test"], desc="Evaluate"):
        cost_value = helper.training.eval_step(parameters, predictor, batch)
        aggregated_cost += cost_value
    test_loss = np.sqrt(aggregated_cost / len(dataset["test"]))
    logging.info(f"Test loss {test_loss}")

    checkpoint_path = pathlib.Path().resolve() / pathlib.Path(filename).stem
    qnnf.save_checkpoints(parameters, tx, step=n_epochs, ckpt_dir=str(checkpoint_path))

    # report
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    report = {
        MetricName.DATE: date_time,
        MetricName.N_QUBITS: num_qubits,
        MetricName.TEST_LOSS: test_loss,
        MetricName.N_GATES: num_gates,
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
