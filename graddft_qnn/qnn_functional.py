import grad_dft as gd
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

    def _regularize_grid_jax(self, grid, num_qubits, grid_data):
        """
        Pure-JAX 'nearest bin' regularization:
        - Build an m x m x m regular grid (m = cube_root(2**num_qubits))
        - Map each irregular coord to a voxel index
        - Scatter-accumulate values and divide by counts (average per voxel)
        - Works under jit/pmap; no callbacks, no SciPy.
        """
        m = int(np.cbrt(2 ** int(num_qubits)))
        coords = jnp.asarray(grid.coords, dtype=jnp.float64)  # (N,3)
        data = jnp.asarray(grid_data, dtype=jnp.float32)  # (N,) or (N,F)
        N = coords.shape[0]

        # Ensure feature dimension F
        if data.ndim == 1:
            data = data[:, None]  # (N,1)
        F = data.shape[1]

        # Compute voxel indices
        mins = jnp.min(coords, axis=0)  # (3,)
        maxs = jnp.max(coords, axis=0)
        # Avoid /0 when molecule is flat along an axis
        spans = jnp.maximum(maxs - mins, 1e-12)

        norm = (coords - mins) / spans  # [0,1]
        idxf = jnp.floor(norm * (m - 1)).astype(jnp.int32)  # (N,3)
        ix, iy, iz = [jnp.clip(idxf[:, k], 0, m - 1) for k in range(3)]  # each (N,)

        # Ravel voxel index
        ravel = ix * (m * m) + iy * m + iz  # (N,)

        # Scatter-add values and counts
        out_vals = jnp.zeros((m * m * m, F), dtype=jnp.float32)
        out_counts = jnp.zeros((m * m * m, 1), dtype=jnp.float32)

        out_vals = out_vals.at[ravel].add(data)  # sum per voxel
        out_counts = out_counts.at[ravel].add(1.0)  # count per voxel

        # Avoid divide-by-zero: keep empty voxels at 0
        out = jnp.where(out_counts > 0, out_vals / out_counts, 0.0)  # (m^3, F)

        return out.reshape((m, m, m, F)) if F > 1 else out.reshape((m, m, m))

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

        try:
            if isinstance(unscaled_coefficient_inputs, JVPTracer):
                # original behavior for traced JVP in non-pmap paths
                interpolated_charge_density = self._regularize_grid(
                    grid, n_qubits, unscaled_coefficient_inputs.aval.val
                )
            else:
                # original behavior
                interpolated_charge_density = self._regularize_grid(
                    grid, n_qubits, unscaled_coefficient_inputs
                )
        except Exception:
            # under pmap/jit, NumPy/SciPy path may hit tracers -> safe fallback
            interpolated_charge_density = self._regularize_grid_jax(
                grid,
                n_qubits,
                jnp.asarray(unscaled_coefficient_inputs, dtype=jnp.float32),
            )

        try:
            # original flatten & append logic
            for _idx in range(interpolated_charge_density.shape[3]):  # shape (n, 2)
                temp_density = interpolated_charge_density[:, :, :, _idx].flatten()[
                    :, np.newaxis
                ]
                if _idx == 0:
                    downsampled_charge_density = temp_density
                else:
                    downsampled_charge_density = np.append(
                        downsampled_charge_density, temp_density, axis=1
                    )
        except Exception:
            # fallback for pmap/tracer case
            downsampled_charge_density = interpolated_charge_density.reshape(
                -1, interpolated_charge_density.shape[-1]
            )

        # downsampling grid weight
        try:
            # original behavior
            interpolated_grid_weights = self._regularize_grid(
                grid, n_qubits, grid.weights
            )
            downsampled_grid_weights = interpolated_grid_weights.flatten()
        except Exception:
            # safe fallback for pmap/jit tracer cases
            interpolated_grid_weights = self._regularize_grid_jax(
                grid, n_qubits, jnp.asarray(grid.weights, dtype=jnp.float32)
            )
            downsampled_grid_weights = interpolated_grid_weights.reshape(-1)

        # normalization
        num_electron = self.integrate_density_with_weights(
            grid.weights, unscaled_coefficient_inputs
        )
        factor_electron = self.integrate_density_with_weights(
            downsampled_grid_weights, downsampled_charge_density
        )

        tot_volume = jnp.sum(grid.weights)
        factor_volume = jnp.sum(downsampled_grid_weights)

        normalized_grid_weights = downsampled_grid_weights * tot_volume / factor_volume

        normalized_charge_density = (
            downsampled_charge_density
            * num_electron
            / factor_electron
            / tot_volume
            * factor_volume
        )

        if FLAG_STANDARD:  # standardization
            # subtract mean
            mean = jax.numpy.mean(normalized_charge_density)  # noqa F821
            charge_density_centered = normalized_charge_density - mean  # noqa F821

            # divide by standard deviation
            std = jax.numpy.std(normalized_charge_density)  # noqa F821
            charge_density_standardized = charge_density_centered / std
            normalized_charge_density = charge_density_standardized

        normalized_energy_densities = self.normalize_energy_density(
            normalized_charge_density
        )

        # get coefficients
        coefficients = self.coefficients.apply(
            params, normalized_charge_density.sum(axis=1)
        )
        coefficients = coefficients[:, jax.numpy.newaxis]  # shape (xxx, 1)
        coefficients = jnp.concatenate(
            (coefficients, coefficients), axis=1
        )  # shape (xxx, 2)

        # should we bring back normal scale
        xc_energy_density = jnp.einsum(
            "rf,rf->r", coefficients, normalized_energy_densities
        )
        xc_energy_density = abs_clip(xc_energy_density, clip_cte)
        return self._integrate(xc_energy_density, normalized_grid_weights)

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
    def normalize_energy_density(downsampled_charge_density):
        lda_x_e = (
            -3
            / 2
            * (3 / (4 * jnp.pi)) ** (1 / 3)
            * (downsampled_charge_density ** (4 / 3)).sum(axis=1, keepdims=True)
        )
        pw92_c_e = gd.popular_functionals.pw92_c_e(downsampled_charge_density)
        pw92_c_e = jnp.expand_dims(pw92_c_e, axis=1)
        return jnp.concatenate((lda_x_e, pw92_c_e), axis=1)
