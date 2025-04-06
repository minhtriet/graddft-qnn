import grad_dft as gd
from optax import apply_updates
from pyscf import dft, gto


# @partial(jax.jit, static_argnums=(1,))
def train_step(parameters, predictor, batch, opt_state, tx):
    atom_coords = list(zip(batch["symbols"], batch["coordinates"]))
    mol = gto.M(atom=atom_coords, basis="STO-3G")
    mean_field = dft.UKS(mol)
    mean_field.kernel()
    molecule = gd.molecule_from_pyscf(mean_field)

    (cost_value, predicted_energy), grads = gd.simple_energy_loss(
        parameters, predictor, molecule, batch["groundtruth"]
    )
    updates, opt_state = tx.update(grads, opt_state, parameters)
    parameters = apply_updates(parameters, updates)

    return parameters, opt_state, cost_value


def eval_step(parameters, predictor, batch):
    atom_coords = list(zip(batch["symbols"], batch["coordinates"]))
    mol = gto.M(atom=atom_coords, basis="def2-tzvp")
    mean_field = dft.UKS(mol)
    mean_field.kernel()
    molecule = gd.molecule_from_pyscf(mean_field)

    (cost_value, predicted_energy), _ = gd.simple_energy_loss(
        parameters, predictor, molecule, batch["groundtruth"]
    )
    return cost_value
