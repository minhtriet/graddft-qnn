import logging
import pathlib
from dataclasses import dataclass

import grad_dft as gd
import numpy as np
import pennylane as qml
import tqdm
from jax import numpy as jnp
from jax.random import PRNGKey
from optax import adamw

from datasets import DatasetDict
from graddft_qnn import helper
from graddft_qnn.trainer.base_simulator import Simulator

logging.getLogger().setLevel(logging.INFO)


@dataclass
class GradientBasedSimulator(Simulator):
    dev: qml.devices.device_api.Device
    dataset: DatasetDict
    n_epochs: int
    batch_size: int
    learning_rate: int
    eval_per_x_epoch: int
    functional: gd.Functional
    model_filename: str

    def simulate(self) -> tuple[float, list[float], list[float]]:
        key = PRNGKey(42)
        predictor = gd.non_scf_predictor(self.functional)
        # get a sample batch for initialization
        coeff_input = jnp.empty((2 ** len(self.dev.wires),))
        logging.info("Initializing the params")
        parameters = self.functional.coefficients.init(key, coeff_input)
        logging.info("Finished initializing the params")
        tx = adamw(learning_rate=self.learning_rate, weight_decay=1e-5)
        opt_state = tx.init(parameters)
        train_losses = []
        test_losses = []
        train_ds = self.dataset["train"]
        for epoch in range(self.n_epochs):
            train_ds = train_ds.shuffle(seed=42)
            aggregated_train_loss = 0

            for i in tqdm.tqdm(
                range(0, len(train_ds), self.batch_size), desc=f"Epoch {epoch + 1}"
            ):
                batch = train_ds[i : i + self.batch_size]
                if len(batch["symbols"]) < self.batch_size:
                    # drop last batch if len(train_ds) % batch_size > 0
                    continue
                parameters, opt_state, cost_value = helper.training.train_step(
                    parameters, self.predictor, batch, opt_state, self.tx
                )
                aggregated_train_loss += cost_value

            # drop last batch if len(train_ds) % batch_size > 0
            num_train_batch = int(np.floor(len(train_ds) / self.batch_size))
            train_loss = np.sqrt(aggregated_train_loss / num_train_batch)

            logging.info(f"RMS train loss: {train_loss}")
            train_losses.append(train_loss)

            if (epoch + 1) % self.eval_per_x_epoch == 0:
                aggregated_cost = 0
                for batch in tqdm.tqdm(
                    self.dataset["test"],
                    desc=f"Evaluate per {self.eval_per_x_epoch} epoch",
                ):
                    cost_value = helper.training.eval_step(
                        parameters, self.predictor, batch
                    )
                    aggregated_cost += cost_value
                test_loss = np.sqrt(aggregated_cost / len(self.dataset["test"]))
                test_losses.append({epoch: test_loss})
                logging.info(f"Test loss: {test_loss}")

        logging.info("Start evaluating")
        # test
        aggregated_cost = 0
        for batch in tqdm.tqdm(self.dataset["test"], desc="Evaluate"):
            cost_value = helper.training.eval_step(parameters, predictor, batch)
            aggregated_cost += cost_value
        test_loss = np.sqrt(aggregated_cost / len(self.dataset["test"]))
        logging.info(f"Test loss {test_loss}")

        checkpoint_path = (
            pathlib.Path().resolve() / pathlib.Path(self.model_filename).stem
        )
        self.functional.save_checkpoints(
            parameters, tx, step=self.n_epochs, ckpt_dir=str(checkpoint_path)
        )
        return test_loss, train_losses, test_losses
