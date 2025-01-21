import os.path

import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')
import tensorflow_datasets as tfds

import pathlib
from ase.io import cube
import numpy as np
import tensorflow as tf
import jax.numpy as jnp


# Step 1: Read the Cube file and extract the grid



# Example usage:
cube_file_paths = os.path.join("graddft_qnn", "cube")
batch_size = 2