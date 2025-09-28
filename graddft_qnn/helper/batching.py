import grad_dft as gd
import jax
import jax.numpy as jnp
from pyscf import dft, gto


def preprocess_molecule(symbols, coordinates):
    """
    Make a list of molecules
    """
    atom_coords = list(zip(symbols, coordinates))
    mol = gto.M(atom=atom_coords, basis="def2-tzvp")
    mf = dft.UKS(mol)
    mf.grids.build()
    mf.kernel()
    return gd.molecule_from_pyscf(mf)


def _stack_pytrees(pytrees):
    """Stack a list of pytrees along a new leading axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *pytrees)


def batch_to_jax(batch, n_devices: int):
    mol_list = [
        preprocess_molecule(syms, coords)
        for syms, coords in zip(batch["symbols"], batch["coordinates"])
    ]

    targets = jnp.asarray(batch["groundtruth"], dtype=jnp.float64)

    mol_batched = _stack_pytrees(mol_list)
    targets = jnp.reshape(targets, (n_devices,))

    return mol_batched, targets
