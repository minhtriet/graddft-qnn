import grad_dft as gd
import jax
import jax.numpy as jnp
import numpy as np
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
    # Molecule can compute the density matrix.
    rho = jnp.clip(molecule.density(), a_min=clip_cte)
    # Now we can implement the LDA energy density equation in the paper.
    lda_e = (
        -3
        / 2
        * (3 / (4 * jnp.pi)) ** (1 / 3)
        * (rho ** (4 / 3)).sum(axis=1, keepdims=True)
    )
    # For simplicity, we do not include the exchange polarization correction
    # check function exchange_polarization_correction in functional.py
    # The output of features must be an Array of dimension n_grid x n_features.
    return lda_e


def load_config(conf_file_name: str) -> tuple:
    with open("config.yaml") as file:
        data = yaml.safe_load(file)
        if "QBITS" not in data:
            raise KeyError("YAML file must contain 'QBITS' key")
        num_qubits = data["QBITS"]
        size = np.cbrt(2**num_qubits)
        assert size.is_integer()
        size = int(size)
        n_epochs = data["TRAINING"]["N_EPOCHS"]
        learning_rate = data["TRAINING"]["LEARNING_RATE"]
        momentum = data["TRAINING"]["MOMENTUM"]
        num_gates = data["N_GATES"]
        eval_per_x_epoch = data["TRAINING"]["EVAL_PER_X_EPOCH"]
        batch_size = data["TRAINING"]["BATCH_SIZE"]
        check_group = data["CHECK_GROUP"]
        assert (
            isinstance(num_gates, int) or num_gates == "full"
        ), f"N_GATES must be integer or 'full', got {num_gates}"
        group: list = data["GROUP"]
        xc_functional_name = data["XC_FUNCTIONAL"]
    return (
        num_qubits,
        size,
        n_epochs,
        learning_rate,
        momentum,
        eval_per_x_epoch,
        batch_size,
        group,
        xc_functional_name,
        check_group,
        num_gates,
    )
