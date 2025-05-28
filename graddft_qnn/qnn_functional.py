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
        with open("config.yaml") as file:
            data = yaml.safe_load(file)
            if "QBITS" not in data:
                raise KeyError("YAML file must contain 'QBITS' key")
            n_qubits = data["QBITS"]

        # taking 2**n_qubits indices
        indices = jnp.round(
            jnp.linspace(0, unscaled_coefficient_inputs.shape[0], 2 ** n_qubits)
        ).astype(jnp.int32)

        # down sampling and normalization
        unnormalized_coefficient_inputs = QNNFunctional.compute_slice_sums(
            unscaled_coefficient_inputs, indices)
        grid_weights = QNNFunctional.compute_slice_sums(
            grid.weights, indices)
        num_electron = jnp.einsum("r,r->",
            grid.weights, unscaled_coefficient_inputs)
        unnormalized_num_electron = jnp.einsum("r,r->",
            grid_weights, unnormalized_coefficient_inputs)
        coefficient_inputs = unnormalized_coefficient_inputs * num_electron / unnormalized_num_electron

        # obtain xc energy
        coefficients = self.coefficients.apply(params, coefficient_inputs)[:, jax.numpy.newaxis]  # shape (xxx, 1)
        densities = QNNFunctional.energy_densities_LDA(coefficient_inputs)
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
        sums = cumsum[ends] - cumsum[starts]
        return sums

    @staticmethod
    def energy_densities_LDA(rho):
        return -3 / 2 * (3 / (4 * jnp.pi)) ** (1 / 3) * (rho ** (4 / 3))[:, jnp.newaxis]