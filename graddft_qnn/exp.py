import tensorflow_datasets as tfds
import pathlib

import graddft_qnn.cube_dataset

mnist_data = tfds.load('cube_dataset', batch_size=-1, data_dir=pathlib.Path('graddft_qnn') / "cube_dataset")
