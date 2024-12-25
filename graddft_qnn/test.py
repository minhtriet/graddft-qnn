from optax import apply_updates
from pyscf import gto, dft
import grad_dft as gd
from jax.random import PRNGKey
from jax import numpy as jnp
from tqdm import tqdm

from dft_qnn import DFTQNN

from qnn_functional import QNNFunctional

from optax import adam


def coefficient_inputs(molecule: gd.Molecule, *_, **__):
    rho = molecule.density()
    # kinetic = molecule.kinetic_density()
    # todo IMPORTANT down sample the 3d image. n quit to encode 2^n amplitude
    # todo entry point must be jax array, not a scalar or anything
    # todo experiment with jax before this
    return rho
    # return jnp.concatenate((rho, kinetic), axis=1)


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


def coefficients_(_, rhoinputs):
    r"""
    Instance is an instance of the class Functional or NeuralFunctional.
    rhoinputs is the input to the neural network, in the form of an array.
    localfeatures represents the potentials e_\theta(r).

    The output of this function is the energy density of the system.
    """

    x = nn.Dense(features=out_features)(rhoinputs)
    x = nn.LayerNorm()(x)
    return sigmoid(x)


if __name__ == "__main__":
    dft_qnn = DFTQNN("config.yaml")  # todo start simpler, make sure input output shape

    mol = gto.M(atom=[["H", (0, 0, 0)], ["F", (0, 0, 1.1)]], basis="def2-tzvp")
    mean_field = dft.UKS(mol)
    ground_truth_energy = mean_field.kernel()

    # Then we can use the following function to generate the molecule object
    HF_molecule = gd.molecule_from_pyscf(mean_field)
    coefficients = dft_qnn.circuit()

    nf = QNNFunctional(coefficients, energy_densities, coefficient_inputs)
    key = PRNGKey(42)
    cinputs = coefficient_inputs(HF_molecule)

    # Init the params
    params = nf.init(key, cinputs)
    # Start the training

    learning_rate = 0.01
    momentum = 0.9
    tx = adam(learning_rate=learning_rate, b1=momentum)
    opt_state = tx.init(params)

    E = nf.energy(params, HF_molecule)

    predictor = gd.non_scf_predictor(nf)

    # training loop
    # todo n_epochs from yaml instead
    n_epochs = 3
    for iteration in tqdm(range(n_epochs), desc="Training epoch"):
        (cost_value, predicted_energy), grads = gd.simple_energy_loss(
            params, predictor, HF_molecule, ground_truth_energy
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
