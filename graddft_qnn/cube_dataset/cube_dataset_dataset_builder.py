import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ase.io import cube


class CubeDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cube_dataset dataset.
    Running tfds build --manual_dir=data will generate a dataset at
    ~/tensorflow_datasets/cube_dataset/1.0.0
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Register into https://example.org/login to get the data. Place the `data.zip`
    file in the `manual_dir/`.
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(cube_dataset): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "grid": tfds.features.Tensor(
                        shape=(80, 80, 80), dtype=tf.float32
                    ),
                    "gs_energy": tfds.features.Tensor(
                        shape=(), dtype=tf.float32
                    ),  # A scalar value
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.manual_dir

        return {
            "train": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for cube_file in path.glob("*.cube"):
            with open(cube_file) as f:
                xyz = cube.read_cube(f, read_data=True, program=None, verbose=False)
                yield (
                    cube_file.name,
                    {
                        'grid': xyz['data'].astype(np.float32),
                        "gs_energy": 1,
                    },
                )
