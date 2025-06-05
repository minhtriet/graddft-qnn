import grad_dft as gd
import numpy as np
import pytest
from pyscf import dft, gto

from graddft_qnn.qnn_functional import QNNFunctional


@pytest.fixture
def qnnf():
    qnnf = QNNFunctional(
        coefficients=None, energy_densities=None, coefficient_inputs=None
    )
    return qnnf


@pytest.fixture
def molecule():
    atom_coords = [("H", [-1.1527353, 0.0, 0.0]), ("H", [1.1527353, 0.0, 0.0])]
    mol = gto.M(atom=atom_coords, basis="def2-tzvp")
    mean_field = dft.UKS(mol)
    mean_field.kernel()
    molecule = gd.molecule_from_pyscf(mean_field)
    return molecule


def test_downsampling_weight(qnnf, molecule):
    before_weight = sum(molecule.grid.weights)
    new_grid = qnnf.grid_weight_downsampling(molecule.grid)
    assert np.isclose(new_grid.grid.weights, before_weight)


def test_downsampling_charge_density(qnnf, molecule):
    rho = molecule.density()
    before_density = sum(np.sum(molecule.density(), 1) * molecule.grid.weights)
    downsampled_grid = qnnf.grid_weight_downsampling(molecule.grid)
    downsampled_density = qnnf.charge_density_downsampling(rho)
    after_density = sum(np.sum(downsampled_density, 1) * downsampled_grid)
    assert np.isclose(before_density, after_density)
