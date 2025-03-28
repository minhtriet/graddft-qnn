# ruff: noqa
import grad_dft as gd
from pyscf import dft, gto

# Define the geometry of the molecule
mol = gto.M(
    atom=[["H", (0, 0, 0)], ["H", (0, 0, 1)]], basis="def2-tzvp", charge=0, spin=0
)
mf = dft.UKS(mol)
ground_truth_energy = mf.kernel()

# Then we can use the following function to generate the molecule object
HH_molecule = gd.molecule_from_pyscf(mf)

import jax.numpy as jnp


def coefficient_inputs(molecule: gd.Molecule, *_, **__):
    rho = molecule.density()
    kinetic = molecule.kinetic_density()
    return jnp.concatenate((rho, kinetic), axis=1)


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


from flax import linen as nn
from jax.nn import sigmoid

out_features = 1


def coefficients(instance, rhoinputs):
    r"""
    Instance is an instance of the class Functional or NeuralFunctional.
    rhoinputs is the input to the neural network, in the form of an array.
    localfeatures represents the potentials e_\theta(r).

    The output of this function is the energy density of the system.
    """

    x = nn.Dense(features=out_features)(rhoinputs)
    x = nn.LayerNorm()(x)
    return sigmoid(x)


nf = gd.NeuralFunctional(coefficients, energy_densities, coefficient_inputs)

from jax.random import PRNGKey

key = PRNGKey(42)
cinputs = coefficient_inputs(HH_molecule)
params = nf.init(key, cinputs)

E = nf.energy(params, HH_molecule)
print("Neural functional energy with random parameters is", E)


from optax import adam

learning_rate = 0.01
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
opt_state = tx.init(params)

predictor = gd.non_scf_predictor(nf)

from optax import apply_updates
from tqdm import tqdm

n_epochs = 20
for iteration in tqdm(range(n_epochs), desc="Training epoch"):
    (cost_value, predicted_energy), grads = gd.simple_energy_loss(
        params, predictor, HH_molecule, ground_truth_energy
    )
    print(
        "Iteration",
        iteration,
        "Predicted energy:",
        predicted_energy,
        "Cost value:",
        cost_value,
    )
    updates, opt_state = tx.update(grads, opt_state, params)
    params = apply_updates(params, updates)

nf.save_checkpoints(params, tx, step=n_epochs)
