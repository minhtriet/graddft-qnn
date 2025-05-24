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
        **kwargs,
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
        numerator = jnp.sum(unscaled_coefficient_inputs, axis=0)
        indices = jnp.round(
            jnp.linspace(0, unscaled_coefficient_inputs.shape[0], 2**n_qubits)
        ).astype(jnp.int32)  # taking 2**n_qubits indices
        # because the size of the grid is bigger than the actual input feedable
        # to the QNN, we do downsample here by summing the negihbors together
        unnormalized_coefficient_inputs = QNNFunctional.compute_slice_sums(
            unscaled_coefficient_inputs, indices
        )
        denominator = jnp.sum(unnormalized_coefficient_inputs)
        coefficient_inputs = unnormalized_coefficient_inputs / denominator * numerator

        grid_numerator = jnp.sum(grid.weights)
        grid_weights = QNNFunctional.compute_slice_sums(grid.weights, indices)
        grid_weights = grid_weights / jnp.sum(grid_weights) * grid_numerator

        # densities: (xxx, 2)
        densities = unscaled_densities[indices]

        # subtract mean
        mean = jax.numpy.mean(coefficient_inputs)
        coefficient_centered = coefficient_inputs - mean

        # divide by standard deviation
        std = jax.numpy.std(coefficient_inputs)
        coefficient_standardized = coefficient_centered / std

        # bar_plot_jvp(coefficient_standardized, "column_chart_standard.png")

        coefficients = self.coefficients.apply(params, coefficient_standardized)
        coefficients *= std
        coefficients += mean

        coefficients = coefficients[:, jax.numpy.newaxis]  # shape (xxx, 1)

        # should we bring back normal scale
        xc_energy_density = jnp.einsum("rf,rf->r", coefficients, densities)
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
