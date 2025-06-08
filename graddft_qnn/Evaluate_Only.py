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
from jax.random import PRNGKey
from optax import adamw

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

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    with open("config.yaml") as file:
        data = yaml.safe_load(file)
        if "QBITS" not in data:
            raise KeyError("YAML file must contain 'QBITS' key")
        num_qubits = data["QBITS"]
        test_num = data.get("TEST_NUM", 5)
        size = np.cbrt(2**num_qubits)
        assert size.is_integer()
        size = int(size)
        test_num = data["TRAINING"]["TEST_NUM"]
        n_epochs = data["TRAINING"]["N_EPOCHS"]
        learning_rate = data["TRAINING"]["LEARNING_RATE"]
        momentum = data["TRAINING"]["MOMENTUM"]
        num_gates = data["N_GATES"]
        check_group = data["CHECK_GROUP"]
        group: list = data["GROUP"]
        group_str_rep = "]_[".join(group)[:230]
        xc_functional_name = data["XC_FUNCTIONAL"]
        dev = qml.device("default.qubit", wires=num_qubits)

    filename = f"ansatz_{num_qubits}_{group_str_rep}_qubits"

    if "naive" not in group[0].lower():
        if pathlib.Path(f"{filename}.pkl").exists():
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
        dft_qnn = NaiveDFTQNN(dev, num_gates)

    qnnf = QNNFunctional(
        coefficients=dft_qnn,
        energy_densities=helper.initialization.energy_densities,
        coefficient_inputs=helper.initialization.coefficient_inputs,
    )

    checkpoint_path = pathlib.Path(filename) / f"checkpoint_{n_epochs}"
    tx = adamw(learning_rate=learning_rate, b1=momentum)
    state = qnnf.load_checkpoint(tx, ckpt_dir=str(checkpoint_path))
    parameters = state.params

    predictor = gd.non_scf_predictor(qnnf)

    if pathlib.Path("datasets/h2_dataset").exists():
        dataset = DatasetDict.load_from_disk(pathlib.Path("datasets/h2_dataset"))
    else:
        dataset = H2MultibondDataset.get_dataset()

    logging.info("Start evaluating")
    test_losses = []
    for repeat in range(test_num):
        aggregated_cost = 0
        for batch in tqdm.tqdm(dataset["test"], desc=f"Test run {repeat+1}/{test_num}"):
            cost_value = helper.training.eval_step(parameters, predictor, batch)
            aggregated_cost += cost_value
        test_loss = np.sqrt(aggregated_cost / len(dataset["test"]))
        test_losses.append(test_loss)
        logging.info(f"Test loss (run {repeat+1}): {test_loss}")

    avg_test_loss = float(np.mean(test_losses))
    logging.info(f"Average Test loss over {test_num} runs: {avg_test_loss}")

    # Write report
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    report = {
        MetricName.DATE: date_time,
        MetricName.N_QUBITS: num_qubits,
        MetricName.TEST_LOSSES: test_losses,
        MetricName.N_GATES: num_gates,
        MetricName.N_MEASUREMENTS: "prob",
        MetricName.GROUP_MEMBER: group,
        MetricName.EPOCHS: n_epochs,
        MetricName.LEARNING_RATE: learning_rate,
        MetricName.BATCH_SIZE: data["TRAINING"]["BATCH_SIZE"],
        "TEST_NUM": test_num,
    }

    if pathlib.Path("report_eval.json").exists():
        with open("report_eval.json") as f:
            try:
                history_report = json.load(f)
            except json.decoder.JSONDecodeError:
                history_report = []
    else:
        history_report = []

    history_report.append(report)
    with open("report_eval.json", "w") as f:
        json.dump(history_report, f)
    pd.DataFrame(history_report).to_excel("report_eval.xlsx")
