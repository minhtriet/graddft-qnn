import grad_dft as gd
import jax
from optax import apply_updates
from pyscf import dft, gto


def train_step(parameters, predictor, batch, opt_state, tx):
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

    avg_cost = sum(cost_values) / len(batch)
    return parameters, opt_state, avg_cost


def train_step_mps(parameters, predictor, batch):
    for example_id in range(len(batch["symbols"])):
        atom_coords = list(
            zip(batch["symbols"][example_id], batch["coordinates"][example_id])
        )
        mol = gto.M(atom=atom_coords, basis="def2-tzvp")
        mean_field = dft.UKS(mol)
        mean_field.kernel()
        molecule = gd.molecule_from_pyscf(mean_field)

        optimize_result = jax.scipy.optimize.minimize(
            fun=gd.simple_energy_loss,
            x0=parameters["params"]["theta"],
            method="BFGS",
            args=(predictor, molecule, batch["groundtruth"][example_id]),
        )

    return optimize_result


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
