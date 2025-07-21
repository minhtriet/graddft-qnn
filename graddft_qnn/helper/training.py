from functools import partial
from typing import Callable

import grad_dft as gd
import jax
from optax import apply_updates
from pyscf import dft, gto
from jax import value_and_grad
from jaxtyping import PyTree, Float

from grad_dft import (
    Molecule,
    Solid,
)

def train_step(parameters, predictor, batch, opt_state, tx):
    """
    :param parameters:
    :param predictor:
    :param batch:
    :param opt_state:
    :param tx:
    :return:
    avg loss of the batch
    """
    grads = []
    cost_values = []
    for example_id in range(len(batch["symbols"])):
        atom_coords = list(
            zip(batch["symbols"][example_id], batch["coordinates"][example_id])
        )
        mol = gto.M(atom=atom_coords, basis="def2-tzvp")
        mean_field = dft.UKS(mol)
        mean_field.kernel()
        molecule = gd.molecule_from_pyscf(mean_field)
        (cost_value, predicted_energy), grad = gd.simple_energy_loss(
            parameters, predictor, molecule, batch["groundtruth"][example_id]
        )
        grads.append(grad)
        cost_values.append(cost_value)

    final_grad = {"params": dict()}
    for param in grads[0]["params"]:
        final_grad["params"][param] = sum(
            grads[i]["params"][param] for i in range(len(grads))
        ) / len(grads)
    updates, opt_state = tx.update(final_grad, opt_state, parameters)
    parameters = apply_updates(parameters, updates)

    # divide by the length of elements in batch, not the number of keys in batch
    avg_cost = sum(cost_values) / len(cost_values)
    print(f"Parameters: {parameters}, grad {final_grad}, avg cost {avg_cost}")

    return parameters, opt_state, avg_cost


def train_step_non_grad(parameters, predictor, batch, tx):
    """
    :param parameters:
    :param predictor:
    :param batch:
    :param opt_state:
    :param tx:
    :return:
    avg loss of the batch
    """
    atom_coords = list(
        zip(batch["symbols"][0], batch["coordinates"][0])
    )
    mol = gto.M(atom=atom_coords, basis="def2-tzvp")
    mean_field = dft.UKS(mol)
    mean_field.kernel()
    molecule = gd.molecule_from_pyscf(mean_field)
    atoms_out = predictor(parameters, molecule)
    result = simple_energy_loss_non_grad(
        e_predict=float(atoms_out.energy),
        truth_energy=batch["groundtruth"][0],
    )
    return result


def eval_step(parameters, predictor, batch):
    atom_coords = list(zip(batch["symbols"], batch["coordinates"]))
    mol = gto.M(atom=atom_coords, basis="def2-tzvp")
    mean_field = dft.UKS(mol)
    mean_field.kernel()  # pass max_cycles / increase iteration
    molecule = gd.molecule_from_pyscf(mean_field, scf_iteration=200)

    (cost_value, predicted_energy), _ = gd.simple_energy_loss(
        parameters, predictor, molecule, batch["groundtruth"]
    )
    return cost_value


def simple_energy_loss_non_grad(
    e_predict: float,
    truth_energy: float,
):
    r"""
    Computes the loss for a single molecule

    Parameters
    ----------
    e_predict: Float
        The predicted energy value from the model
    truth_energy: Float
        The energy value we are training against
    """
    diff = e_predict - truth_energy
    return diff**2


@partial(value_and_grad, has_aux=True, argnums=[0])
def simple_energy_loss(params: PyTree,
    compute_energy: Callable,
    atoms: Molecule | Solid,
    truth_energy: Float,
    ):
    r"""
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
