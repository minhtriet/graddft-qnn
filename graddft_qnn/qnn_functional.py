from collections.abc import Callable

import jax
import yaml
from grad_dft import Functional, Solid, abs_clip
from grad_dft.molecule import Grid, Molecule
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree, Scalar


class QNNFunctional(Functional):
    def xc_energy(
        self,
        params: PyTree,
        grid: Grid,
        unscaled_coefficient_inputs: Float[Array, "(n,n)"],
        unscaled_densities: Float[Array, "(n,n)"],
        clip_cte: float = 1e-30,
    ) -> Scalar:
        """
        :param params:
        :param grid:
        :param coefficient_inputs:  grid coeff inputs
        :param densities: grid densities
        :param clip_cte:
        :return:
        """
        # down-sampling coeff_inputs, densities
        # rescale densities and grid_weights
        with open("config.yaml") as file:
            data = yaml.safe_load(file)
            if "QBITS" not in data:
                raise KeyError("YAML file must contain 'QBITS' key")
            n_qubits = data["QBITS"]
        # unscaled_coeff_inputs: (xxx, 2)

        numerator = jnp.sum(unscaled_coefficient_inputs, axis=0)
        indices = jnp.round(
            jnp.linspace(0, unscaled_coefficient_inputs.shape[0], 2**n_qubits)
        ).astype(jnp.int32)  # taking 2**n_qubits indices
        unnormalized_coefficient_inputs = unscaled_coefficient_inputs[indices]
        denominator = jnp.sum(unnormalized_coefficient_inputs, axis=0)
        coefficient_inputs = unnormalized_coefficient_inputs / denominator * numerator

        grid_nominator = jnp.sum(grid.weights)
        grid_weights = grid.weights[indices]
        grid_weights = grid_weights / jnp.sum(grid_weights) * grid_nominator

        # densities: (xxx, 2)
        densities = unscaled_densities[indices]

        coefficients = self.coefficients.apply(params, coefficient_inputs)
        coefficients = coefficients[:, jax.numpy.newaxis]  # shape (xxx, 1)
        xc_energy_density = jnp.einsum("rf,rf->r", coefficients, densities)
        xc_energy_density = abs_clip(xc_energy_density, clip_cte)
        return self._integrate(xc_energy_density, grid_weights)  # was grid.weights

    @jax.jit
    def loss_fn(
        params,
        data,
        compute_energy: Callable,
        atoms: Molecule | Solid,
        truth_energy: Float,
    ):
        atoms_out = compute_energy(params, atoms)
        E_predict = atoms_out.energy
        diff = E_predict - truth_energy
        return diff**2, E_predict
