from flax import linen as nn
from jax.nn import sigmoid
from jax.nn import gelu
import jax
from optax import apply_updates
from optax import adam
from jax.random import PRNGKey

from datasets import DatasetDict
import sys
import pathlib
import logging
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from pyscf import gto, dft
import os
from graddft_qnn.cube_dataset.h2_multibond import H2MultibondDataset
from graddft_qnn.qnn_functional import QNNFunctional
import grad_dft as gd
from grad_dft.popular_functionals import pw92_densities
from datetime import datetime
import json
import pandas as pd
import jax.numpy as jnp
import yaml


mol = gto.M(atom=[["H", (0, 0, 0)], ["H", (0, 0, 1)]], basis="def2-tzvp", charge=0, spin=0)
mf = dft.UKS(mol)
ground_truth_energy = mf.kernel()

HH_molecule = gd.molecule_from_pyscf(mf)

density = HH_molecule.density()
grid = HH_molecule.grid

density_sum = np.sum(density, axis=1, keepdims=True)
weighted = density_sum * grid.weights[:, jnp.newaxis]
total = np.sum(weighted)

print(total)