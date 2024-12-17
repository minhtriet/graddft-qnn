from optax import apply_updates
from pyscf import gto, dft
import grad_dft as gd
from jax.random import PRNGKey
from jax import numpy as jnp
from tqdm import tqdm

from dft_qnn import DFTQNN

from optax import adam


def coefficient_inputs(molecule: gd.Molecule, *_, **__):
    rho = molecule.density()
    # todo kinetic seems not written in E_{xc}[\rho] = \int \c_\theta[\rho](r) \cdot e[\rho](r)dr
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


if __name__ == "__main__":
    dft_qnn = DFTQNN("config.yaml")

    mol = gto.M(atom=[["H", (0, 0, 0)], ["F", (0, 0, 1.1)]], basis="def2-tzvp", charge=0, spin=1)
    mean_field = dft.UKS(mol)
    ground_truth_energy = mean_field.kernel()

    # Then we can use the following function to generate the molecule object
    HF_molecule = gd.molecule_from_pyscf(mean_field)
    coefficients = dft_qnn.circuit()  # todo add phi and theta here

    nf = gd.Functional(coefficients, energy_densities, coefficient_inputs)
    key = PRNGKey(42)
    cinputs = coefficient_inputs(HF_molecule)

    # todo change interface to autograd?

    # Init the params
    params = nf.init(key, cinputs)
    # Start the training

    learning_rate = 0.01
    momentum = 0.9
    tx = adam(learning_rate=learning_rate, b1=momentum)
    opt_state = tx.init(params)
    key = PRNGKey(42)
    cinputs = coefficient_inputs(HF_molecule)
    params = nf.init(key, cinputs)

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
