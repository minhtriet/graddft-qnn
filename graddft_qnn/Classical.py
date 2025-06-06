import json
import logging
import os
import pathlib
from datetime import datetime

import grad_dft as gd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import yaml
from flax import linen as nn
from jax.nn import sigmoid
from jax.nn import gelu
from jax.random import PRNGKey
from optax import adam, apply_updates
from pyscf import dft, gto

from datasets import DatasetDict
from graddft_qnn.cube_dataset.h2_multibond import H2MultibondDataset
from graddft_qnn.qnn_functional import QNNFunctional

with open("config.yaml") as file:
    data = yaml.safe_load(file)
num_qubits = data["QBITS"]
n_epochs = data["TRAINING"]["N_EPOCHS"]
learning_rate = data["TRAINING"]["LEARNING_RATE"]
momentum = data["TRAINING"]["MOMENTUM"]
eval_per_x_epoch = data["TRAINING"]["EVAL_PER_X_EPOCH"]
batch_size = data["TRAINING"]["BATCH_SIZE"]

# Define the geometry of the molecule
mol = gto.M(
    atom=[["H", (0, 0, 0)], ["H", (0, 0, 1)]], basis="def2-tzvp", charge=0, spin=0
)
mf = dft.UKS(mol)
ground_truth_energy = mf.kernel()

# Then we can use the following function to generate the molecule object
HH_molecule = gd.molecule_from_pyscf(mf)


def coefficient_inputs(molecule: gd.Molecule, *_, **__):
    rho = molecule.density()
    return rho


def energy_densities(molecule: gd.Molecule, clip_cte: float = 1e-30, *_, **__):
    r"""Auxiliary function to generate the features of LSDA."""
    # Molecule can compute the density matrix.
    rho = jnp.clip(molecule.density(), a_min=clip_cte)
    # Now we can implement the LDA energy density equation in the paper.
    lda_e = -3/2 * (3/(4*jnp.pi)) ** (1/3) * (rho**(4/3)).sum(axis = 1, keepdims = True)
    #pw92_corr_e = pw92_densities(molecule, clip_cte)
    # For simplicity we do not include the exchange polarization correction
    # check function exchange_polarization_correction in functional.py
    # The output of features must be an Array of dimension n_grid x n_features.
    # print(f"LDA Energy Density - Shape: {lda_e.shape}, Size: {lda_e.size}")
    return lda_e


# test_inputs=coefficient_inputs(HH_molecule)
# test_densities=energy_densities(HH_molecule)

squash_offset = 1e-4
layer_widths = [16] * 2
out_features = 1
sigmoid_scale_factor = 2.0
activation = gelu


class NeuralCoeff(nn.Module):
    layer_widths: list[int]
    out_features: int

    @nn.compact
    def __call__(self, x):
        if x.ndim == 1:
            x = x[:, jnp.newaxis]  # Convert shape (batch,) → (batch, 1)
        squash_offset = 1e-4

        x = jnp.log(jnp.abs(x) + squash_offset)
        x = nn.Dense(self.layer_widths[0])(x)
        x = jnp.tanh(x)

        for width in self.layer_widths:
            res = x
            x = nn.Dense(width)(x)
            x = x + res
            x = nn.LayerNorm()(x)
            x = jax.nn.gelu(x)

        x = nn.Dense(self.out_features)(x)
        x = nn.LayerNorm()(x)
        return jax.nn.sigmoid(x).squeeze(-1)


coefficients = NeuralCoeff(layer_widths, out_features)


nf = QNNFunctional(
    coefficients=coefficients,
    energy_densities=energy_densities,
    coefficient_inputs=coefficient_inputs,
)

key = PRNGKey(42)

input_shape = (2**num_qubits, 1)
random_input = jax.random.normal(key, shape=input_shape)
params = nf.coefficients.init(key, random_input)


def params_size(params):
    """Calculate the size of parameters, including nested structures."""
    total_size = 0

    def count_params(p):
        nonlocal total_size
        if isinstance(p, dict):
            for value in p.values():
                count_params(value)
        elif isinstance(p, jnp.ndarray):
            total_size += p.size

    count_params(params)
    return total_size


num_params = params_size(params)

# E = nf.energy(params, HH_molecule)
# print("Neural functional energy with random parameters is", E)


tx = adam(learning_rate=learning_rate, b1=momentum)
opt_state = tx.init(params)

predictor = gd.non_scf_predictor(nf)

# start training
if pathlib.Path("datasets/h2_dataset").exists():
    dataset = DatasetDict.load_from_disk(pathlib.Path("datasets/h2_dataset"))
else:
    dataset = H2MultibondDataset.get_dataset()
    dataset.save_to_disk("datasets/h2_dataset")


# train
train_losses = []
train_losses_batch = []
test_losses = []
train_ds = dataset["train"]
for epoch in range(n_epochs):
    train_ds = train_ds.shuffle(seed=42)
    aggregated_train_loss = 0

    for i in tqdm.tqdm(range(0, len(train_ds), batch_size), desc=f"Epoch {epoch + 1}"):
        batch = train_ds[i : i + batch_size]
        if len(batch["symbols"]) < batch_size:
            # drop last batch if len(train_ds) % batch_size > 0
            continue
        cost_values = []
        for example_id in range(len(batch["symbols"])):
            atom_coords = list(
                zip(batch["symbols"][example_id], batch["coordinates"][example_id])
            )
            mol = gto.M(atom=atom_coords, basis="def2-tzvp")
            mean_field = dft.UKS(mol)
            # mean_field.xc = 'wB97M-V'
            # mean_field.nlc = 'VV10'
            mean_field.kernel()
            molecule = gd.molecule_from_pyscf(mean_field)

            (cost_value, predicted_energy), grads = gd.simple_energy_loss(
                params, predictor, molecule, batch["groundtruth"][example_id]
            )
            updates, opt_state = tx.update(grads, opt_state, params)
            params = apply_updates(params, updates)

            cost_values.append(cost_value)
            # avg_cost = sum(cost_values) / len(batch) #30
            avg_cost = sum(cost_values)

        aggregated_train_loss += avg_cost
        train_losses_batch.append(np.sqrt(avg_cost / len(batch["symbols"])))
    num_train_batch = int(np.floor(len(train_ds)/batch_size))
    train_loss = np.sqrt(aggregated_train_loss /num_train_batch)
    logging.info(f"RMS train loss: {train_loss}")
    train_losses.append(train_loss)

    if (epoch + 1) % eval_per_x_epoch == 0:
        aggregated_cost = 0
        for batch in tqdm.tqdm(
            dataset["test"], desc=f"Evaluate per {eval_per_x_epoch} epoch"
        ):
            "from helper.training.eval_step"
            atom_coords = list(zip(batch["symbols"], batch["coordinates"]))
            mol = gto.M(atom=atom_coords, basis="def2-tzvp")
            mean_field = dft.UKS(mol)
            # mean_field.xc = 'wB97M-V'
            # mean_field.nlc = 'VV10'
            mean_field.kernel()  # pass max_cycles / increase iteration
            molecule = gd.molecule_from_pyscf(mean_field, scf_iteration=200)

            (cost_value, predicted_energy), _ = gd.simple_energy_loss(
                params, predictor, molecule, batch["groundtruth"]
            )
            aggregated_cost += cost_value

        test_loss = np.sqrt(aggregated_cost / len(dataset["test"]))
        test_losses.append({epoch: test_loss})
        logging.info(f"Test loss: {test_loss}")


def get_unique_filename(base_filename):
    filename = base_filename
    file_counter = 1

    while os.path.exists(filename):
        base, ext = os.path.splitext(base_filename)
        filename = f"{base}_{file_counter}{ext}"
        file_counter += 1

    return filename


# Plot binding energy
distances = np.arange(0.2, 5.0, 0.1)  # Distance range from 1 to 5 with step 0.3
E_predicts = []

for distance in tqdm.tqdm(distances, desc="Calculating Binding Energy"):
    # Create molecule with the specified distance
    mol = gto.M(
        atom=[["H", (0, 0, 0)], ["H", (0, 0, distance)]],
        basis="def2-tzvp",
        unit="Angstrom",
    )
    mean_field = dft.UKS(mol)
    # mean_field.xc = 'wB97M-V'
    # mean_field.nlc = 'VV10'

    ground_truth_energy = mean_field.kernel()
    molecule = gd.molecule_from_pyscf(mean_field)

    (cost_value, predicted_energy), _ = gd.simple_energy_loss(
        params, predictor, molecule, ground_truth_energy
    )
    E_predicts.append(predicted_energy)


distances = np.array(distances)
E_predicts = np.array(E_predicts)


plt.figure(figsize=(10, 6))
plt.plot(
    distances,
    E_predicts,
    marker="o",
    linestyle="-",
    color="blue",
    label="Predicted Energy",
)
plt.xlabel("Distance between H atoms (Å)")
plt.ylabel("Binding Energy (Hartree)")
plt.title("Binding Energy Curve")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

# Define the filename
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
base_filename = os.path.join(output_dir, "binding_energy_Classical.png")
final_filename = get_unique_filename(base_filename)

# Save the plot as PNG with high resolution
plt.savefig(final_filename, dpi=300)
print(f"Plot saved as: {final_filename}")

plt.show()


# Function to handle JAX array serialization
def to_serializable(obj):
    if isinstance(obj, jnp.ndarray):
        return obj.tolist()
    if isinstance(obj, (jnp.float32, jnp.float64)):
        return float(obj)
    if isinstance(obj, (jnp.int32, jnp.int64)):
        return int(obj)
    return obj


# Report generation
now = datetime.now()
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

# Create report dictionary
report = {
    "DATE": date_time,
    "N_QUBITS": num_qubits,
    "TEST_LOSS": test_loss,
    "EPOCHS": n_epochs,
    "TRAIN_LOSSES": train_losses,
    "TRAIN_LOSSES_BY_BATCH": train_losses_batch,
    "TEST_LOSSES": test_losses,
    "LEARNING_RATE": learning_rate,
    "BATCH_SIZE": batch_size,
    "Number of Parameters": num_params,
    "Momentum": momentum,
    "eval_per_x_epoch": eval_per_x_epoch,
}

# Check if the report.json file exists
report_path = "classical_report.json"
if pathlib.Path(report_path).exists():
    with open(report_path) as f:
        try:
            history_report = json.load(f)
        except json.decoder.JSONDecodeError:
            history_report = []
else:
    history_report = []

history_report.append(report)

# Save to JSON with proper conversion
with open(report_path, "w") as f:
    json.dump(history_report, f, indent=4, default=to_serializable)

# Save to Excel
df = pd.DataFrame(history_report)
df.to_excel("classical_report.xlsx", index=False)
