import logging
import pathlib
import sys

import grad_dft as gd
import numpy as np
import pennylane as qml
import yaml
from jax import numpy as jnp
from jax.random import PRNGKey
from jaxtyping import PyTree
from optax import adam, apply_updates
from pyscf import dft, gto
from tqdm import tqdm

from graddft_qnn import custom_gates
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.io.ansatz_io import AnsatzIO
from graddft_qnn.qnn_functional import QNNFunctional
from graddft_qnn.unitary_rep import O_h

logging.getLogger().setLevel(logging.INFO)


def coefficient_inputs(molecule: gd.Molecule, *_, **__):
    rho = molecule.density()
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


def simple_energy_loss(
    params: PyTree,
    compute_energy,  #:  Callable,
    atoms,  #: #Union[Molecule, Solid],
    truth_energy,  #: #Float,
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


if __name__ == "__main__":
    with open("config.yaml") as file:
        data = yaml.safe_load(file)
        if "QBITS" not in data:
            raise KeyError("YAML file must contain 'QBITS' key")
        num_qubits = data["QBITS"]
        n_epochs = data["TRAINING"]["N_EPOCHS"]
        learning_rate = data["TRAINING"]["LEARNING_RATE"]
        momentum = data["TRAINING"]["MOMENTUM"]
        dev = qml.device("default.qubit", wires=num_qubits)

    size = np.cbrt(2 ** len(dev.wires))
    assert size.is_integer()
    size = int(size)

    ansatz_io = AnsatzIO()
    filename = f"ansatz_{num_qubits}_qubits.txt"
    if pathlib.Path(filename).exists():
        gates_gen = ansatz_io.read_from_file(filename)
        logging.info(f"Loaded ansatz generator from {filename}")
    else:
        gates_gen = DFTQNN.gate_design(len(dev.wires), O_h.C2_group(size, True))
        ansatz_io.write_to_file(filename, gates_gen)
    measurement_expvals = [
        custom_gates.generate_operators(measurement) for measurement in gates_gen
    ]
    dft_qnn = DFTQNN(dev, gates_gen, measurement_expvals)

    mol = gto.M(atom=[["H", (0, 0, 0)], ["F", (0, 0, 1.1)]], basis="def2-tzvp")
    mean_field = dft.UKS(mol)
    ground_truth_energy = mean_field.kernel()
    HF_molecule = gd.molecule_from_pyscf(mean_field)
    # there is a charge density in grad_dft, downsize and set it back to the downsized properties

    key = PRNGKey(42)
    coeff_input = coefficient_inputs(HF_molecule)
    indices = jnp.round(jnp.linspace(0, coeff_input.shape[0], 2**num_qubits)).astype(
        jnp.int32
    )
    coeff_input = coeff_input[indices]
    logging.info("Initializing the params")
    parameters = dft_qnn.init(key, coeff_input)
    logging.info("Finished initializing the params")

    nf = QNNFunctional(
        coefficients=dft_qnn,
        energy_densities=energy_densities,
        coefficient_inputs=coefficient_inputs,
    )
    tx = adam(learning_rate=learning_rate, b1=momentum)
    opt_state = tx.init(parameters)

    predictor = gd.non_scf_predictor(nf)

    for iteration in tqdm(range(n_epochs), desc="Training epoch", file=sys.stdout):
        (cost_value, predicted_energy), grads = gd.simple_energy_loss(
            parameters, predictor, HF_molecule, ground_truth_energy
        )
        print(
            "Iteration",
            iteration,
            "Predicted energy:",
            predicted_energy,
            "Cost value:",
            cost_value,
            "Grad: ",
            (jnp.max(grads["params"]["theta"]), jnp.min(grads["params"]["theta"])),
        )
        updates, opt_state = tx.update(grads, opt_state, parameters)
        parameters = apply_updates(parameters, updates)
"""
1. automate the twirling, measurement + ansatz
1. do 2 3 molecules, compare to classical
2. increase finess of grids
3. log the xc_energy + total_energy
4. regularization (params 0 - 2pi)
# todo calculate the number of symmetric equivalent pixel rather than all pixels
"""
