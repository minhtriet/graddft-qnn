"""
This will
1. read the config specified in config.yaml
2. load the checkpoint associated with that config
3. do annotation
"""

import logging
import pathlib

import jax
import numpy as np
import pennylane as qml
import yaml

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
        if "naive" not in group[0].lower():
            group_str_rep = "]_[".join(group)[:230]
            group_matrix_reps = [getattr(O_h, gr)(size, False) for gr in group]
            if (check_group) and (not is_group(group_matrix_reps, group)):
                raise ValueError("Not forming a group")
        xc_functional_name = data["XC_FUNCTIONAL"]
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

    checkpoint_path = pathlib.Path().resolve() / pathlib.Path(filename).stem
    state = qnnf.load_checkpoint(ckpt_dir=str(checkpoint_path))
