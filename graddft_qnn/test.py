import jax
import optax
import yaml
from jaxtyping import PyTree
from pyscf import gto, dft
import grad_dft as gd
from jax.random import PRNGKey
import pennylane as qml
from jax import numpy as jnp
from tqdm import tqdm
from optax import apply_updates

from dft_qnn import DFTQNN

from qnn_functional import QNNFunctional

from optax import adam


def coefficient_inputs(molecule: gd.Molecule, *_, **__):
    rho = molecule.density()
    # todo IMPORTANT down sample the 3d image. n quit to encode 2^n amplitude
    # todo entry point must be jax array, not a scalar or anything
    # todo experiment with jax before this
    return rho


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

def simple_energy_loss(params: PyTree,
    compute_energy,#:  Callable,
    atoms,#: #Union[Molecule, Solid],
    truth_energy,#: #Float,
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

@jax.jit
def update_step(opt, params, opt_state, data, targets):
    loss_val, grads = jax.value_and_grad(simple_energy_loss, argnums=[1, 2])(cinputs, dft_qnn.phi, dft_qnn.theta, predictor, HF_molecule,
                                                       ground_truth_energy)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

@jax.jit
def optimization_jit(params, data, targets, print_training=False):
    opt = optax.adam(learning_rate=0.3)

    opt_state = opt.init(params)
    args = (params, opt_state, data, targets, print_training)
    (params, opt_state, _, _, _) = jax.lax.fori_loop(0, 100, update_step, args)

    return params


if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        data = yaml.safe_load(file)
        if "QBITS" not in data:
            raise KeyError("YAML file must contain 'QBITS' key")
        num_qubits = data["QBITS"]
        dev = qml.device("default.qubit", wires=num_qubits)
    dft_qnn = DFTQNN(dev)  # todo start simpler, make sure input output shape

    mol = gto.M(atom=[["H", (0, 0, 0)], ["F", (0, 0, 1.1)]], basis="def2-tzvp")
    mean_field = dft.UKS(mol)
    ground_truth_energy = mean_field.kernel()
    HF_molecule = gd.molecule_from_pyscf(mean_field)

    key = PRNGKey(42)
    # coeff_input = dim_reduction(coefficient_inputs(HF_molecule))
    coeff_input = coefficient_inputs(HF_molecule)
    parameters = dft_qnn.init(key, coeff_input)

    nf = QNNFunctional(coefficients=dft_qnn,
                       energy_densities=energy_densities,
                       coefficient_inputs=coefficient_inputs)

    # Start the training
    learning_rate = 0.1
    momentum = 0.9
    n_epochs = 20

    tx = adam(learning_rate=learning_rate, b1=momentum)
    opt_state = tx.init(parameters)

    predictor = gd.non_scf_predictor(nf)
    # pca normalization is correct or not, becasue only coeff inputs is normalized, still have grid, density

    for iteration in tqdm(range(n_epochs), desc="Training epoch"):
        (cost_value, predicted_energy), grads = gd.simple_energy_loss(
            parameters, predictor, HF_molecule, ground_truth_energy
        )
        print("Iteration", iteration, "Predicted energy:", predicted_energy, "Cost value:", cost_value)
        updates, opt_state = tx.update(grads, opt_state, parameters)
        parameters = apply_updates(parameters, updates)

    nf.save_checkpoints(parameters, tx, step=n_epochs)