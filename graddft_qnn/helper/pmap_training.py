import grad_dft as gd
import jax
import optax

from .batching import batch_to_jax


def train_step(parameters, predictor, batch, opt_state, tx):
    # Make per-device batch
    n_devices = jax.device_count()
    mol_jax, targets = batch_to_jax(batch, n_devices=n_devices)

    # Single-device loss+grad
    def loss_and_grad_one(p, mol, y):
        (loss, _), grad = gd.simple_energy_loss(p, predictor, mol, y)
        return loss, grad

    #  pmap wrapper that does cross-device mean ON DEVICE
    def per_device(p, mol, y):
        loss, grad = loss_and_grad_one(p, mol, y)
        loss = jax.lax.pmean(loss, axis_name="i")
        grad = jax.lax.pmean(grad, axis_name="i")
        return loss, grad

    loss_grad_pmap = jax.pmap(
        per_device,
        in_axes=(None, 0, 0),  # params broadcast; mol & targets sharded
        out_axes=(None, None),  # return host values (already reduced)
        axis_name="i",
    )

    # run pmap
    loss_value, grads = loss_grad_pmap(parameters, mol_jax, targets)

    # Optax update on host (single grads pytree)
    updates, opt_state = tx.update(grads, opt_state, parameters)
    new_params = optax.apply_updates(parameters, updates)

    return new_params, opt_state, loss_value
