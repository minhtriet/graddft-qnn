from typing import Callable, Union

import grad_dft as gd
import jax
from jaxlib.xla_extension import ArrayImpl
from grad_dft.molecule import Grid, Molecule
from jax import numpy as jnp
from grad_dft import abs_clip, Solid
from jaxtyping import Array, PyTree, Scalar, Float
import pcax
from standard_scaler import StandardScaler


class QNNFunctional(gd.Functional):
    """
    This functional uses more jax API than flax
    """

    def dim_reduction(self, original_array: ArrayImpl):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(original_array)
        state = pcax.fit(X_scaled, n_components=3)
        X_pca = pcax.transform(state, X_scaled)
        return X_pca

    def xc_energy(
            self,
            params: PyTree,
            grid: Grid,
            coefficient_inputs: Float[Array, "grid cinputs"],
            densities: Float[Array, "grid densities"],
            clip_cte: float = 1e-30,
            **kwargs
    ) -> Scalar:
        coefficients = self.apply(params, coefficient_inputs, **kwargs)
        # coefficients = coefficients[:coefficient_inputs.shape[0]]
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