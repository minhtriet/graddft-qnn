import jax
import numpy as np
import scipy
import yaml
from grad_dft import NeuralFunctional, abs_clip
from grad_dft.molecule import Grid
from jax import numpy as jnp
from jax._src.interpreters.ad import JVPTracer
from jaxtyping import Array, Float, PyTree, Scalar


class QNNFunctional(NeuralFunctional):
    def _regularize_grid(self, grid: Grid, num_qubits, grid_data):
        # 1. grid.coordinates are not sorted
        # 2. grid.coordinates are not regular grid
        # Due to this irregularity, we cannot immediately use RegularGridInterpolator
        x = grid.coords[:, 0]
        y = grid.coords[:, 1]
        z = grid.coords[:, 2]
        per_axis_dimension = int(np.cbrt(2**num_qubits))

        new_x, new_y, new_z = np.mgrid[
            np.min(x) : np.max(x) : per_axis_dimension * 1j,
            np.min(y) : np.max(y) : per_axis_dimension * 1j,
            np.min(z) : np.max(z) : per_axis_dimension * 1j,
        ]

        interpolated = scipy.interpolate.griddata(
            grid.coords,
            grid_data,
            (new_x, new_y, new_z),
            method="nearest",
        ).astype(jnp.float32)
        return interpolated

    def xc_energy(
        self,
        params: PyTree,
        grid: Grid,
        unscaled_coefficient_inputs: Float[Array, "(n,n)"],  # noqa: F821
        unscaled_densities: Float[Array, "(n,n)"],  # noqa: F821
        clip_cte: float = 1e-30,
        FLAG_STANDARD: bool = False,
        **kwargs,
    ) -> Scalar:
        """
        It is ok to pass a "standardized" coefficient_inputs to the neural net.
        It is not ok to and the mean and divide by the standard deviation for the output
        of the neural net.
        Potentially we can also remove the standardization of the coefficient_inputs
        before passing to the neural net. This is sometimes a technique used in
        classical NN but it is not clear if we need it here.

        :param params:
        :param grid:
        :param unscaled_coefficient_inputs: energy density
        :param unscaled_densities: grid densities
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

        # downsampling charge density
        if isinstance(unscaled_coefficient_inputs, JVPTracer):
            interpolated_charge_density = self._regularize_grid(
                grid, n_qubits, unscaled_coefficient_inputs.aval.val
            ).flatten()
        else:
            interpolated_charge_density = self._regularize_grid(
                grid, n_qubits, unscaled_coefficient_inputs
            ).flatten()

        # downsampling grid weight
        interpolated_grid_weights = self._regularize_grid(
            grid, n_qubits, grid.weights
        ).flatten()

        # downsampling charge density
        interpolated_energy_densities = self.downsampling_energy_density(
            interpolated_charge_density
        )
        interpolated_energy_densities = interpolated_energy_densities[
            :, jax.numpy.newaxis
        ]

        # normalization
        num_electron = self.integrate_density_with_weights(
            grid.weights, unscaled_coefficient_inputs
        )
        factor_electron = self.integrate_density_with_weights(
            interpolated_grid_weights, interpolated_charge_density
        )

        tot_volume = jnp.sum(grid.weights)
        factor_volume = jnp.sum(interpolated_grid_weights)

        normalized_grid_weights = interpolated_grid_weights * tot_volume / factor_volume
        normalized_charge_density = (
            interpolated_charge_density
            * num_electron
            / factor_electron
            / tot_volume
            * factor_volume
        )

        if FLAG_STANDARD: # standardization
            # subtract mean
            mean = jax.numpy.mean(normalized_charge_density)  # noqa F821
            charge_density_centered = normalized_charge_density - mean  # noqa F821

            # divide by standard deviation
            std = jax.numpy.std(normalized_charge_density)  # noqa F821
            charge_density_standardized = charge_density_centered / std
            normalized_charge_density = charge_density_standardized

        # get coefficients
        coefficients = self.coefficients.apply(params, normalized_charge_density)
        coefficients = coefficients[:, jax.numpy.newaxis]  # shape (xxx, 1)

        # should we bring back normal scale
        xc_energy_density = jnp.einsum(
            "rf,rf->r", coefficients, interpolated_energy_densities
        )
        xc_energy_density = abs_clip(xc_energy_density, clip_cte)
        return self._integrate(
            xc_energy_density, normalized_grid_weights
        )  # was grid.weights

    @staticmethod
    def compute_slice_sums(X, indices):
        """
        with indices = [0, x, y ...], return [sum(X[0:x]), sum(X[x:y])]
        """
        # Compute cumulative sum, prepending 0 to handle start=0
        cumsum = jnp.concatenate([jnp.array([0]), jnp.cumsum(X)])
        starts = jnp.array(indices)
        ends = jnp.concatenate([indices[1:], jnp.array([len(X)])])
        sums = cumsum[ends] - cumsum[starts]
        return sums

    @staticmethod
    def integrate_density_with_weights(grid_weights, density):
        """
        Computes elementwise multiplication of summed density and grid weights,
        then returns the total sum.

        :param grid_weights: Grid weights (shape: [n])
        :param density: Densities (shape: [n, f])
        :return: Scalar result of ∑_i (grid[i] * ∑_j density[i, j])
        """
        if density.ndim == 1:
            density = density[:, jnp.newaxis]  # Reshape (n,) → (n, 1)

        density_sum = jnp.sum(density, axis=1, keepdims=True)  # shape (n, 1)
        weighted = density_sum * grid_weights[:, jnp.newaxis]  # shape (n, 1)
        return jnp.sum(weighted)

    @staticmethod
    def downsampling_energy_density(charge_density):
        r"""Auxiliary function to generate the features of LSDA."""
        # Now we can implement the LDA energy density equation in the paper.
        lda_e = (
            -3
            / 2
            * (3 / (4 * jnp.pi)) ** (1 / 3)
            * (charge_density ** (4 / 3))
        )
        # For simplicity, we do not include the exchange polarization correction
        # check function exchange_polarization_correction in functional.py
        # The output of features must be an Array of dimension n_grid x n_features.
        return lda_e