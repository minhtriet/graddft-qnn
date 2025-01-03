from typing import Callable, Union

import grad_dft as gd
import jax
from grad_dft.molecule import Grid, Molecule
from jax import numpy as jnp
from grad_dft import abs_clip, Solid
from jaxtyping import Array, PyTree, Scalar, Float
import flax.linen as nn


class QNNFunctional(gd.Functional):
    """
    This functional uses more jax API than flax
    """

    def xc_energy(
            self,
            params: PyTree,
            grid: Grid,
            coefficient_inputs: Float[Array, "grid cinputs"],
            densities: Float[Array, "grid densities"],
            clip_cte: float = 1e-30,
    ) -> Scalar:
        coefficients = self.coefficients.apply(params, coefficient_inputs)
        # todo ask for confirm
        coefficients = coefficients[:coefficient_inputs.shape[0]]  # shape: (xxx)
        coefficients = coefficients[:, jax.numpy.newaxis]  # shape (xxx, 1)
        # coeffs have norm of 1, and  we are multiplying with very small number here, should we normalize density for the sake of the neural net optimization, or we want to obey the physics semantic?
        # look at the 3d image, what we have is a molecule. A large part is 0
        xc_energy_density = jnp.einsum("rf,rf->r", coefficients, densities)
        xc_energy_density = abs_clip(xc_energy_density, clip_cte)
        return self._integrate(xc_energy_density, grid.weights)

    @jax.jit
    def loss_fn(params, data,
        compute_energy: Callable,
        atoms: Union[Molecule, Solid],
        truth_energy: Float,
    ):
        atoms_out = compute_energy(params, atoms)
        E_predict = atoms_out.energy
        diff = E_predict - truth_energy
        return diff ** 2, E_predict