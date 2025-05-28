import jax
import yaml
from grad_dft import NeuralFunctional, abs_clip
from grad_dft.molecule import Grid

# from helper.visualization import bar_plot_jvp
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree, Scalar


class QNNFunctional(NeuralFunctional):
    def xc_energy(
        self,
        params: PyTree,
        grid: Grid,
        unscaled_coefficient_inputs: Float[Array, "(n,n)"],  # noqa: F821
        unscaled_densities: Float[Array, "(n,n)"],  # noqa: F821
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
        # bar_plot_jvp(unscaled_coefficient_inputs, "column_chart_og.png")
        indices = jnp.round(
            jnp.linspace(0, unscaled_coefficient_inputs.shape[0], 2**n_qubits)
        ).astype(jnp.int32)  # taking 2**n_qubits indices
        # because the size of the grid is bigger than the actual input feedable
        # to the QNN, we do downsample here by summing the negihbors together

        coefficient_inputs = QNNFunctional.compute_slice_sums(
            unscaled_coefficient_inputs, indices)
        grid_weights = QNNFunctional.compute_slice_sums(
            grid.weights, indices)

        #Pysical Constraints
        N0 = QNNFunctional.integrate_density_with_weights(grid.weights, unscaled_coefficient_inputs)
        N_down = QNNFunctional.integrate_density_with_weights(grid_weights, coefficient_inputs)

        #Scaling
        coefficient_inputs = coefficient_inputs * N0 / N_down

        #Re-define energy density from down scaled charge density
        densities = QNNFunctional.energy_densities_LDA(coefficient_inputs)

        # obtain coefficients with parameters
        coefficients = self.coefficients.apply(
            params, coefficient_inputs
        )[:, jax.numpy.newaxis] # shape (xxx, 1)

        # obtain xc energy
        xc_energy_density = jnp.einsum("rf,rf->r", coefficients, densities)
        xc_energy_density = abs_clip(xc_energy_density, clip_cte)
        return self._integrate(xc_energy_density, grid_weights)

    @staticmethod
    def compute_slice_sums(X, indices):
        """
        with indices = [0, x, y ...], return [sum(X[0:x]), sum(X[x:y])]
        """
        # Compute cumulative sum, prepending 0 to handle start=0
        cumsum = jnp.concatenate([jnp.array([0]), jnp.cumsum(X)])
        starts = jnp.array(indices)
        ends = jnp.concatenate([indices[1:], jnp.array([len(X)])])
        downsampled = cumsum[ends] - cumsum[starts]
        return downsampled / jnp.sum(downsampled) * jnp.sum(X) #return it after rescailing

    @staticmethod
    def integrate_density_with_weights(
        grid_weights: Float[Array, "n"],
        density: Float[Array, "n f"],
    ) -> Scalar:
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
    def energy_densities_LDA(rho):
        return -3 / 2 * (3 / (4 * jnp.pi)) ** (1 / 3) * (rho ** (4 / 3))[:, jnp.newaxis]