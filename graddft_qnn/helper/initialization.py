import grad_dft as gd
import jax
import jax.numpy as jnp
import yaml


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
    return jnp.sum(rho, 1)


@jax.jit
def energy_densities(molecule: gd.Molecule, clip_cte: float = 1e-30, *_, **__):
    r"""Auxiliary function to generate the features of LSDA."""
    with open("config.yaml") as file:
        data = yaml.safe_load(file)
        if "COR_ENERGY_DENSITY" not in data:
            raise KeyError("YAML file must contain 'COR_ENERGY_DENSITY' key")
        cor_energy_density = data["COR_ENERGY_DENSITY"]
    # Molecule can compute the density matrix.
    rho = jnp.clip(molecule.density(), a_min=clip_cte)
    # Now we can implement the LDA energy density equation in the paper.
    lda_x_e = (
        -3
        / 2
        * (3 / (4 * jnp.pi)) ** (1 / 3)
        * (rho ** (4 / 3)).sum(axis=1, keepdims=True)
    )
    # For simplicity, we do not include the exchange polarization correction
    # check function exchange_polarization_correction in functional.py
    # The output of features must be an Array of dimension n_grid x n_features.
    if not cor_energy_density:
        return lda_x_e
    else:
        pw92_c_e = gd.popular_functionals.pw92_densities(molecule, clip_cte)
        return jnp.concatenate((lda_x_e, pw92_c_e), axis=1)