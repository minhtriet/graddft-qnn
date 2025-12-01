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
from optax import adamw

from datasets import DatasetDict
from graddft_qnn import helper
from graddft_qnn.cube_dataset.h2_multibond import H2MultibondDataset
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.helper.initialization import batched

# from graddft_qnn.helper import training
from graddft_qnn.io.ansatz_io import AnsatzIO
from graddft_qnn.naive_dft_qnn import NaiveDFTQNN
from graddft_qnn.qnn_functional import QNNFunctional
from graddft_qnn.unitary_rep import O_h, is_group

logging.getLogger().setLevel(logging.INFO)
np.random.seed(42)
key = PRNGKey(42)


def reorder(lst, step):
    """
    Reorder a list by taking elements in column-wise strides.

    Given a list ``lst`` and an integer ``step``, this function returns a new
    list formed by starting at each offset ``0`` through ``step - 1`` and
    collecting elements spaced ``step`` apart.

    For example, with ``lst = [1,2,3,4,5,6,7,8,9,10]`` and ``step = 3``,
    the function produces::

        [1, 4, 7, 2, 5, 8, 3, 6, 9]

    because it takes:
    - elements at indices 0, 3, 6, ...
    - then indices 1, 4, 7, ...
    - then indices 2, 5, 8, ...
    stopping when indices exceed the list length.

    Parameters
    ----------
    lst : list
        The input list to reorder.
    step : int
        The stride length used to collect column-wise elements. Must be
        a positive integer.

    Returns
    -------
    list
        A new list containing the reordered elements.
    """
    result = []
    n = len(lst)
    for offset in range(step):
        i = offset
        while i < n:
            result.append(lst[i])
            i += step
    return result


def draw_circuit_mpl(dev, gate_gens):
    """
    As the qml.draw() doesn't seem to support gates from qml.exp(), we do it ourself
    """
    drawer = qml.drawer.MPLDrawer(n_layers=10, wire_map={i: i for i in dev.wires})
    layer = 1
    last_wire = 0
    for gate_gen in gate_gens:
        if (last_wire > list(gate_gen.wires.labels)[-1]) or (
            list(gate_gen.wires.labels)[0] < last_wire < list(gate_gen.wires.labels)[-1]
        ):
            layer += 1
        drawer.box_gate(
            layer=layer, text=str(gate_gen)[25:50], wires=list(gate_gen.wires.labels)
        )
        last_wire = list(gate_gen.wires.labels)[-1]

    drawer.fig.savefig("circuit.png")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    with open("config.yaml") as file:
        data = yaml.safe_load(file)
        num_qubits = data["NETWORK"]["WIRES"]
        network_type = data["NETWORK"]["TYPE"]
        group_qubits_size = data["GROUP"]["QUBITS"]  # for group
        size = np.cbrt(2**group_qubits_size)
        assert size.is_integer()
        size = int(size)
        n_epochs = data["TRAINING"]["N_EPOCHS"]
        learning_rate = data["TRAINING"]["LEARNING_RATE"]
        momentum = data["TRAINING"]["MOMENTUM"]
        num_gates = data["N_GATES"]
        eval_per_x_epoch = data["TRAINING"]["EVAL_PER_X_EPOCH"]
        batch_size = data["TRAINING"]["BATCH_SIZE"]
        check_group = data["CHECK_GROUP"]
        flag_meanfield = data["FLAG_MEANFIELD"]
        assert (
            isinstance(num_gates, int) or num_gates == "full"
        ), f"N_GATES must be integer or 'full', got {num_gates}"
        full_measurements = "prob"
        group: list = data["GROUP"]["MEMBERS"]
        group_str_rep = "]_[".join(group)[:230]
        if "naive" not in group:
            group_matrix_reps = [getattr(O_h, gr)(size, False) for gr in group]
            if (check_group) and (not is_group(group_matrix_reps, group)):
                raise ValueError("Not forming a group")
        xc_functional_name = data["XC_FUNCTIONAL"]
        dev = qml.device("default.qubit", wires=num_qubits)

    # define the QNN
    filename = f"ansatz_{num_qubits}_{group_str_rep}_{group_qubits_size}_qubits"
    if "naive" not in group:
        if pathlib.Path(f"{filename}.pkl").exists():
            gates_gens = AnsatzIO.read_from_file(filename)
            logging.info(f"Loaded ansatz generator from {filename}")
        else:
            gates_gens = []
            if network_type == "qcnn":
                # a hack: we assume the size of the gate is 3 qubits
                # convolution 1, can be > 1 layer
                for starting_wire in range(0, 3):
                    for batch in batched(
                        range(starting_wire, num_qubits), group_qubits_size
                    ):
                        gates_gens.extend(
                            DFTQNN.gate_design(
                                [
                                    getattr(O_h, gr)(size, True, starting_wire=batch[0])
                                    for gr in group
                                ],
                                wires=batch,
                            )
                        )
                # pooling 1
                # not implemented yet
                gates_gens = reorder(gates_gens, step=3)
            elif network_type == "qnn":
                for batch in batched(range(num_qubits), group_qubits_size):
                    gates_gens.extend(
                        DFTQNN.gate_design(
                            [
                                getattr(O_h, gr)(size, True, starting_wire=batch[0])
                                for gr in group
                            ],
                            wires=batch,
                        )
                    )
            AnsatzIO.write_to_file(filename, gates_gens)
        gates_gens = gates_gens[: 2**num_qubits]
        draw_circuit_mpl(dev, gates_gens)

        gates_indices = sorted(np.random.choice(len(gates_gens), num_gates))

        dft_qnn = DFTQNN(
            dev, gates_gens, gates_indices, network_type=data["NETWORK"]["TYPE"]
        )
    else:
        dft_qnn = NaiveDFTQNN(dev, num_gates)

    # get a sample batch for initialization
    coeff_input = jnp.empty((2 ** len(dev.wires),))
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
    # tx = jaxopt.ScipyMinimize(fun=training.simple_energy_loss, method="COBYLA")

    tx = adamw(learning_rate=learning_rate, weight_decay=1e-5)
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
            if len(batch["symbols"]) < batch_size:
                # drop last batch if len(train_ds) % batch_size > 0
                continue
            #
            # ```python
            parameters, opt_state, cost_value = helper.training.train_step(
                parameters, predictor, batch, opt_state, tx, flag_meanfield
            )
            # ```
            # but now we use jaxopt.ScipyMinimize, so we need to modify this, while
            # the training step should be more agnostic to the optimizer used.
            #
            # cost_value = helper.training.train_step_non_grad(
            #     parameters, predictor, batch, tx
            # )
            aggregated_train_loss += cost_value

        # drop last batch if len(train_ds) % batch_size > 0
        num_train_batch = int(np.floor(len(train_ds) / batch_size))
        train_loss = np.sqrt(aggregated_train_loss / num_train_batch)

        logging.info(f"RMS train loss: {train_loss}")
        train_losses.append(train_loss)

        if (epoch + 1) % eval_per_x_epoch == 0:
            aggregated_cost = 0
            for batch in tqdm.tqdm(
                dataset["test"], desc=f"Evaluate per {eval_per_x_epoch} epoch"
            ):
                cost_value = helper.training.eval_step(
                    parameters, predictor, batch, flag_meanfield
                )
                aggregated_cost += cost_value
            test_loss = np.sqrt(aggregated_cost / len(dataset["test"]))
            test_losses.append({epoch: test_loss})
            logging.info(f"Test loss: {test_loss}")

    logging.info("Start evaluating")
    # test
    aggregated_cost = 0
    for batch in tqdm.tqdm(dataset["test"], desc="Evaluate"):
        cost_value = helper.training.eval_step(
            parameters, predictor, batch, flag_meanfield
        )
        aggregated_cost += cost_value
    test_loss = np.sqrt(aggregated_cost / len(dataset["test"]))
    logging.info(f"Test loss {test_loss}")

    if "naive" not in group[0].lower():
        checkpoint_path = pathlib.Path().resolve() / pathlib.Path(filename).stem
        qnnf.save_checkpoints(
            parameters, tx, step=n_epochs, ckpt_dir=str(checkpoint_path)
        )

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
        MetricName.FLAG_MEANFIELD: flag_meanfield,
    }
    if pathlib.Path("rexport.json").exists():
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
