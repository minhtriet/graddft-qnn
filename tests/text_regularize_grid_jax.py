# tests/test_norm_only.py
import numpy as np
import jax.numpy as jnp
import pytest
from graddft_qnn.qnn_functional import QNNFunctional
from grad_dft.molecule import Grid

def make_irregular_grid(n=4096, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform([-1, -0.5, 0], [2, 0.7, 1.5], size=(n, 3))
    weights = rng.uniform(0.1, 2.0, size=(n,)).astype(np.float32)
    return Grid(coords=coords, weights=weights)

def make_density(n=4096, f=2, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, f)).astype(np.float32)


def apply_normalization(grid_weights, downsampled_grid_weights,
                        unscaled_coefficient_inputs, downsampled_charge_density):

    # Same method in QNNFunctional.py
    def integrate_density_with_weights(grid_w, density):
        if density.ndim == 1:
            density = density[:, None]
        density_sum = jnp.sum(density, axis=1, keepdims=True)
        return jnp.sum(grid_w[:, None] * density_sum)

    num_electron   = integrate_density_with_weights(jnp.asarray(grid_weights),
                                                    jnp.asarray(unscaled_coefficient_inputs))
    factor_electron = integrate_density_with_weights(jnp.asarray(downsampled_grid_weights),
                                                     jnp.asarray(downsampled_charge_density))

    tot_volume   = jnp.sum(jnp.asarray(grid_weights))
    factor_volume = jnp.sum(jnp.asarray(downsampled_grid_weights))

    normalized_grid_weights = jnp.asarray(downsampled_grid_weights) * (tot_volume / factor_volume)
    normalized_charge_density = (jnp.asarray(downsampled_charge_density)
                                 * num_electron / factor_electron
                                 / tot_volume * factor_volume)

    return normalized_grid_weights, normalized_charge_density

@pytest.mark.parametrize("use_reg2", [False, True])
def test_normalization_invariants(use_reg2):
    # data
    grid = make_irregular_grid(n=4096, seed=10)
    unscaled_coefficient_inputs = make_density(n=4096, f=2, seed=11)

    # choose regularizer
    qnn = object.__new__(QNNFunctional)  # no __init__, no required args
    n_qubits = 9  # m=8

    regularizer = qnn._regularize_grid_jax if use_reg2 else qnn._regularize_grid

    # downsample charge_density
    interpolated_charge_density = regularizer(grid, n_qubits, unscaled_coefficient_inputs)
    if interpolated_charge_density.ndim == 3:  # (m,m,m) -> (R,1)
        downsampled_charge_density = interpolated_charge_density.reshape(-1, 1)
    else:  # (m,m,m,F) -> (R,F)
        downsampled_charge_density = interpolated_charge_density.reshape(
            -1, interpolated_charge_density.shape[-1]
        )

    # downsample grid weights
    interpolated_grid_weights = regularizer(grid, n_qubits, grid.weights)  # (m,m,m)
    downsampled_grid_weights = interpolated_grid_weights.reshape(-1)

    # apply normalization
    normalized_grid_weights, normalized_charge_density = apply_normalization(
        grid.weights, downsampled_grid_weights,
        unscaled_coefficient_inputs, downsampled_charge_density
    )

    # invariants
    V_orig = jnp.sum(jnp.asarray(grid.weights))
    V_new  = jnp.sum(normalized_grid_weights)
    assert jnp.allclose(V_new, V_orig, atol=1e-6), "Total volume not preserved"

    N_orig = jnp.sum(jnp.asarray(grid.weights)[:, None]
                     * jnp.sum(jnp.asarray(unscaled_coefficient_inputs), axis=1, keepdims=True))
    N_new  = jnp.sum(normalized_grid_weights[:, None]
                     * jnp.sum(normalized_charge_density, axis=1, keepdims=True))
    assert jnp.allclose(N_new, N_orig, atol=1e-5), "Electron count not preserved"
