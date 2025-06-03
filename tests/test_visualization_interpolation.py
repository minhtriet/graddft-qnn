from jax import numpy as jnp
from datasets import DatasetDict
import pathlib
from graddft_qnn.cube_dataset.h2_multibond import H2MultibondDataset
from pyscf import dft, gto, scf
import grad_dft as gd
import jax
import ase.io.cube
import numpy as np

# obtain training data sample of H2
if pathlib.Path("datasets/h2_dataset").exists():
    dataset = DatasetDict.load_from_disk(pathlib.Path("datasets/h2_dataset"))
else:
    dataset = H2MultibondDataset.get_dataset()
    dataset.save_to_disk("datasets/h2_dataset")
train_ds = dataset["train"]
train_ds = train_ds.shuffle(seed=42)
sample_ds = train_ds[0]

# create molecule from training data
atom_coords = list(
    zip(sample_ds["symbols"], sample_ds["coordinates"]))
mol = gto.M(atom=atom_coords, basis="def2-tzvp")
mean_field = dft.UKS(mol)
mean_field.kernel()
molecule = gd.molecule_from_pyscf(mean_field)

# generate original cube file
from pyscf.tools import cubegen
from ase.io import cube
cubegen.density(mol, 'h2o_den.cube', mean_field.make_rdm1())

import scipy
from grad_dft.molecule import Grid
# interpolation
def _regularize_grid(grid: Grid, num_qubits, grid_data):
    # 1. grid.coordinates are not sorted
    # 2. grid.coordinates are not regular grid
    # Due to this irregularity, we cannot immediately use RegularGridInterpolator
    x = grid.coords[:, 0]
    y = grid.coords[:, 1]
    z = grid.coords[:, 2]
    per_axis_dimension = int(np.cbrt(2 ** num_qubits))

    new_x, new_y, new_z = np.mgrid[
                          np.min(x): np.max(x): per_axis_dimension * 1j,
                          np.min(y): np.max(y): per_axis_dimension * 1j,
                          np.min(z): np.max(z): per_axis_dimension * 1j,
                          ]

    interpolated = scipy.interpolate.griddata(
        grid.coords,
        grid_data,
        (new_x, new_y, new_z),
        method="nearest",
    ).astype(jnp.float32)
    return interpolated

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

# downsampling charge density
@jax.jit
def charge_density(molecule: gd.Molecule, *_, **__):
    rho = molecule.density()
    return jnp.sum(rho, 1)
num_qubits = 15
interpolated_charge_density = _regularize_grid(molecule.grid, num_qubits, charge_density(molecule))

# downsampling grid weight
interpolated_grid_weights = _regularize_grid(molecule.grid, num_qubits, molecule.grid.weights)

# normalization
num_electron = integrate_density_with_weights(molecule.grid.weights, charge_density(molecule))
factor_electron = integrate_density_with_weights(interpolated_grid_weights, interpolated_charge_density)

tot_volume = jnp.sum(molecule.grid.weights)
factor_volume = jnp.sum(interpolated_grid_weights)

normalized_grid_weights = interpolated_grid_weights * tot_volume / factor_volume
normalized_charge_density = interpolated_charge_density * num_electron / factor_electron / tot_volume * factor_volume


# downsampled CUBE
from pyscf import __config__
from pyscf.tools.cubegen import Cube

RESOLUTION = getattr(__config__, 'cubegen_resolution', None)
BOX_MARGIN = getattr(__config__, 'cubegen_box_margin', 3.0)
def density_downsampling(mol, outfile, density, nx=2, ny=2, nz=2, resolution=RESOLUTION,
                         margin=BOX_MARGIN):
    from pyscf.pbc.gto import Cell
    cc = Cube(mol, nx, ny, nz, resolution, margin)

    GTOval = 'GTOval'
    if isinstance(mol, Cell):
        GTOval = 'PBC' + GTOval

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    rho = density
    rho = rho.reshape(cc.nx,cc.ny,cc.nz)

    # Write out density to the .cube file
    cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')
    return rho

# generate downsampled cube file
downsampled_axis = int(np.cbrt(2**num_qubits))
density_downsampling(
    mol, "h2o_den_down.cube", normalized_charge_density,
    nx=downsampled_axis, ny=downsampled_axis, nz=downsampled_axis)




stop = 0

# i) charge density visualization of H2 (compute_slice_sum)
# ii) charge density visualization of H2 (interpolating function)
# iii) grid normalization