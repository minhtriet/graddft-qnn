import jax
import numpy as np
import scipy
import yaml
from grad_dft import NeuralFunctional, abs_clip
from grad_dft.molecule import Grid
from jax import numpy as jnp
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

    def grid_weight_downsampling(self, grid: Grid, num_qubits: int):
        # First handle the grid data.
        interpolated = self._regularize_grid(grid, num_qubits, grid.weights)
        reconstructed_values = interpolated.flatten()

        # renormalize
        reconstructed_values = (
            reconstructed_values / jnp.sum(reconstructed_values) * jnp.sum(grid.weights)
        )

        return reconstructed_values

    def charge_density_downsampling(
        self,
        charge_density: Float[Array, "(n,2)"],  # noqa: F821
        grid: Grid,
        num_qubits: int,
        downsampled_grid: list,
    ):
        assert np.allclose(
            charge_density[:, 0], charge_density[:, 1]
        ), "We are supporting only closed shell molecule right now"

        # First handle the grid data.
        interpolated = self._regularize_grid(grid, num_qubits, charge_density[:, 0])
        interpolated = interpolated.flatten()
        # renormalize
        interpolated = interpolated / sum(interpolated) * sum(charge_density[:, 0])

        # make sure the summed charge density is correct
        correction = (
            downsampled_grid * interpolated - charge_density[:, 0] * grid.weights
        )
        interpolated[0] = interpolated[0] - correction / downsampled_grid[0]
        return np.stack((interpolated, interpolated), axis=1)

    def xc_energy(
        self,
        params: PyTree,
        grid: Grid,
        unscaled_coefficient_inputs: Float[Array, "(n,n)"],  # noqa: F821
        unscaled_densities: Float[Array, "(n,n)"],  # noqa: F821
        clip_cte: float = 1e-30,
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

        coefficients_input = self.charge_density_downsampling(  # noqa F841
            unscaled_coefficient_inputs, grid, n_qubits
        )

        self.charge_density_downsampling(unscaled_densities, grid, n_qubits)

        grid_weights = self.grid_weight_downsampling(grid, n_qubits)
        # densities: (xxx, 2)

        # subtract mean
        mean = jax.numpy.mean(coefficient_inputs)  # noqa F821
        coefficient_centered = coefficient_inputs - mean  # noqa F821

        # divide by standard deviation
        std = jax.numpy.std(coefficient_inputs)  # noqa F821
        coefficient_standardized = coefficient_centered / std

        # bar_plot_jvp(coefficient_standardized, "column_chart_standard.png")

        coefficients = self.coefficients.apply(params, coefficient_standardized)

        coefficients = coefficients[:, jax.numpy.newaxis]  # shape (xxx, 1)
        # should we bring back normal scale
        xc_energy_density = jnp.einsum("rf,rf->r", coefficients, unscaled_densities)
        xc_energy_density = abs_clip(xc_energy_density, clip_cte)
        return self._integrate(xc_energy_density, grid_weights)  # was grid.weights

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
