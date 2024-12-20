import grad_dft as gd
from jaxlib.xla_extension import ArrayImpl
from flax import linen as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from grad_dft.molecule import Grid
from jax import numpy as jnp
from grad_dft import abs_clip
from jaxtyping import Array, PyTree, Scalar, Float

class QNNFunctional(gd.Functional):
    @nn.compact
    def __call__(self, coefficient_inputs) -> Scalar:
        r"""Where the functional is called, mapping the density to the energy.
        Expected to be overwritten by the inheriting class.
        Should use the _integrate() helper function to perform the integration.

        Parameters
        ---------
        inputs: inputs to the function f

        Returns
        -------
        Union[Array, Scalar]
        """
        coefficient_inputs = self.dim_reduction(coefficient_inputs)
        return self.coefficients(coefficient_inputs)

    def dim_reduction(self, original_array: ArrayImpl):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(original_array)
        pca = PCA(n_components=1, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
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
        """
        trim the extra elements of :param: coefficient before calling the upper class's method
        """
        coefficients = self.apply(params, coefficient_inputs, **kwargs)
        coefficients = coefficients[:coefficient_inputs.shape[0]]
        xc_energy_density = jnp.einsum("rf,rf->r", coefficients, densities)
        xc_energy_density = abs_clip(xc_energy_density, clip_cte)
        return self._integrate(xc_energy_density, grid.weights)
