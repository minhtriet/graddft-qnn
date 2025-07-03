import os

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["NUMBA_NUM_THREADS"] = "8"
import json
import logging
import pathlib
from datetime import datetime

import jax
import numpy as np
import pandas as pd
import pennylane as qml
import yaml
from evaluate.metric_name import MetricName

from datasets import DatasetDict
from graddft_qnn import helper, unitary_rep
from graddft_qnn.cube_dataset.h2_multibond import H2MultibondDataset
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.dft_tn import DFTTN
from graddft_qnn.io.ansatz_io import AnsatzIO
from graddft_qnn.naive_dft_qnn import NaiveDFTQNN
from graddft_qnn.qnn_functional import QNNFunctional
from graddft_qnn.trainer.gradient_based_simulator import GradientBasedSimulator
from graddft_qnn.trainer.non_grad_based_simulator import NonGradBasedSimulator
from graddft_qnn.unitary_rep import O_h, is_group

logging.getLogger().setLevel(logging.INFO)
np.random.seed(42)

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
        device_config = data["DEVICE"]
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
        dev = qml.device(**device_config, wires=num_qubits)

    # define the QNN
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
        num_gates = min(num_gates, len(gates_gen))
        gates_indices = sorted(np.random.choice(len(gates_gen), num_gates))
    if dev.name == "default.tensor":
        dft_qnn = DFTTN(dev, gates_gen, gates_indices)
    elif "naive" not in group[0].lower():
        dft_qnn = DFTQNN(dev, gates_gen, gates_indices)
    else:
        dft_qnn = NaiveDFTQNN(dev, num_gates)

    # resolve energy density according to user input
    e_density = helper.initialization.resolve_energy_density(xc_functional_name)

    # define the functional
    qnnf = QNNFunctional(
        coefficients=dft_qnn,
        energy_densities=helper.initialization.energy_densities,
        coefficient_inputs=helper.initialization.coefficient_inputs,
    )

    # start training
    if pathlib.Path("datasets/h2_dataset").exists():
        dataset = DatasetDict.load_from_disk(pathlib.Path("datasets/h2_dataset"))
    else:
        dataset = H2MultibondDataset.get_dataset()
        dataset.save_to_disk("datasets/h2_dataset")

    # train
    if dev.name == "default.qubit":
        simulator = GradientBasedSimulator(
            dev,
            dataset,
            n_epochs,
            batch_size,
            learning_rate,
            eval_per_x_epoch,
            qnnf,
            filename,
        )
    else:
        simulator = NonGradBasedSimulator(dev, qnnf, dataset)
    test_loss, train_losses, test_losses = simulator.simulate()

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
