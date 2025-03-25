from typing import Any

import tensorflow as tf
import tensorflow_datasets as tfds
from pyscf import dft, gto


class CubeDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release with PySCF mol objects and energies."}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.molecules = [
            [["H", (0, 0, 0)], ["H", (0, 0, 1)]],
            [["H", (0, 0, 0)], ["F", (0, 0, 1.1)]],
            [["Li", (-1.3365, 0.0, 0.0)], ["Li", (1.3365, 0.0, 0.0)]],
            [["Na", (-1.53945, 0.0, 0.0)], ["Na", (1.53945, 0.0, 0.0)]],
            [
                ["N", (0, 0, 0)],
                ["H", (0, 0, 1.008)],
                ["H", (0.950353, 0, -0.336)],
                ["H", (-0.475176, -0.823029, -0.336)],
            ],
            [
                ["N", (0.0, 1.36627479, -0.21221668)],
                ["N", (0.0, -1.36627479, -0.21221668)],
                ["H", (-0.84470931, 1.89558816, 1.42901383)],
                ["H", (0.84470931, -1.89558816, 1.42901383)],
                ["H", (1.8260461, 1.89558816, 0.05688087)],
                ["H", (-1.8260461, -1.89558816, 0.05688087)],
            ],
        ]

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="Dataset of molecules with PySCF mol objects and DFT energies",
            features=tfds.features.FeaturesDict(
                {
                    "name": tfds.features.Text(),  # Element symbol
                    "groundtruth": tfds.features.Scalar(dtype=tf.float32),
                    "mean_field": tfds.features.Tensor(shape=(), dtype=tf.float32),
                }
            ),
            supervised_keys=("molecule", "energy"),
            homepage="https://example.com",
            citation="""@article{your_citation, ...}""",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        train_size = int(0.8 * len(self.molecules))
        train_mols = self.molecules[:train_size]
        test_mols = self.molecules[train_size:]

        return {
            "train": self._generate_examples(train_mols),
            "test": self._generate_examples(test_mols),
        }

    def _generate_examples(self, molecules: list[list]) -> dict[int, dict[str, Any]]:
        for idx, molecule in enumerate(molecules):
            mol = gto.M(atom=molecule, basis="def2-tzvp")
            mean_field = dft.UKS(mol)
            yield (
                idx,
                {
                    "name": [atom[0] for atom in molecule],
                    "groundtruth": mean_field.kernel(),
                    "mean_field": mean_field,
                },
            )
