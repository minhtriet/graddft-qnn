import grad_dft as gd
import jax
import jax.numpy as jnp


def resolve_energy_density(xc_functional_name: str):
    xc_functional = getattr(gd.popular_functionals, xc_functional_name, None)
    if xc_functional:
        return xc_functional
    else:
        raise ModuleNotFoundError(
            f"Function {xc_functional} does not exist in popular_functionals"
        )


@jax.jit
def coefficient_inputs(molecule: gd.Molecule, *_, **__):
    rho = molecule.density()
    return rho


@jax.jit
def energy_densities(molecule: gd.Molecule, clip_cte: float = 1e-30, *_, **__):
    r"""Auxiliary function to generate the features of LSDA."""
    # Molecule can compute the density matrix.
    rho = jnp.clip(molecule.density(), a_min=clip_cte)
    # Now we can implement the LDA energy density equation in the paper.
    lda_x_e = (
        -3
        / 2
        * (3 / (4 * jnp.pi)) ** (1 / 3)
        * (rho ** (4 / 3)).sum(axis=1, keepdims=True)
    )
    pw92_c_e = gd.popular_functionals.pw92_densities(molecule, clip_cte)
    # For simplicity, we do not include the exchange polarization correction
    # check function exchange_polarization_correction in functional.py
    # The output of features must be an Array of dimension n_grid x n_features.
    return jnp.concatenate((lda_x_e, pw92_c_e), axis=1)