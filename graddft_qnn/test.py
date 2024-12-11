from pyscf import gto, dft
import grad_dft as gd

from jax import numpy as jnp


def coefficient_inputs(molecule: gd.Molecule, *_, **__):
    rho = molecule.density()
    kinetic = molecule.kinetic_density()
    return jnp.concatenate((rho, kinetic), axis = 1)


def energy_densities(molecule: gd.Molecule, clip_cte: float = 1e-30, *_, **__):
    r"""Auxiliary function to generate the features of LSDA."""
    # Molecule can compute the density matrix.
    rho = jnp.clip(molecule.density(), a_min=clip_cte)
    # Now we can implement the LDA energy density equation in the paper.
    lda_e = -3/2 * (3/(4*jnp.pi)) ** (1/3) * (rho**(4/3)).sum(axis = 1, keepdims = True)
    # For simplicity we do not include the exchange polarization correction
    # check function exchange_polarization_correction in functional.py
    # The output of features must be an Array of dimension n_grid x n_features.
    return lda_e


def coefficients(instance, rhoinputs):
    """
    :param instance: an instance of the class Functional.
    :param rhoinputs: input to the neural network, in the form of an array.
    :return:
    """
    x = rhoinputs
    # todo add debugger here
    # todo allow instance to be QNNFunctional, maybe implement that too
    return rhoinputs[0]

# Define the geometry of the molecule and mean-field object
mol = gto.M(atom=[["H", (0, 0, 0)]], basis="def2-tzvp", charge=0, spin=1)
mf = dft.UKS(mol)
mf.kernel()
# Then we can use the following function to generate the molecule object
HF_molecule = gd.molecule_from_pyscf(mf)

nf = gd.Functional(coefficients, energy_densities, coefficient_inputs)

