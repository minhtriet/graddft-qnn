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
# (expand to training dataset)
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

# generate cube file
from pyscf.tools import cubegen
from ase.io import cube
cubegen.density(mol, 'h2o_den.cube', mean_field.make_rdm1())

# read electron density
with open("h2o_den.cube", "r") as file:
    #spacing = cube.read_cube(file, read_data=True, program=None, verbose=False)["spacing"]
    electron_density = cube.read_cube(file, read_data=True, program=None, verbose=False)["data"]
    electron_density = electron_density.reshape(electron_density.size)

# read spacing
with open("h2o_den.cube", "r") as file:
    spacing = cube.read_cube(file, read_data=True, program=None, verbose=False)["spacing"]
    volume = spacing[0,0] * spacing[1,1] * spacing[2,2]
    volumes = np.ones(electron_density.size) * volume


# generate indicies
n_qubits = 18
indices = jnp.round(jnp.linspace(0, electron_density.shape[0], 2**n_qubits)
        ).astype(jnp.int32)  # taking 2**n_qubits indices

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

# downsampling
from pyscf import __config__
from pyscf.tools.cubegen import Cube

RESOLUTION = getattr(__config__, 'cubegen_resolution', None)
BOX_MARGIN = getattr(__config__, 'cubegen_box_margin', 3.0)
downsampled_electron_density = compute_slice_sums(electron_density, indices)
downsampled_volumes = compute_slice_sums(volumes, indices)

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
downsampled_axis = int(np.cbrt(2**n_qubits))
density_downsampling(
    mol, "h2o_den_down.cube", downsampled_electron_density,
    nx=downsampled_axis, ny=downsampled_axis, nz=downsampled_axis)




stop = 0

# i) charge density visualization of H2 (compute_slice_sum)
# ii) charge density visualization of H2 (interpolating function)
# iii) grid normalization