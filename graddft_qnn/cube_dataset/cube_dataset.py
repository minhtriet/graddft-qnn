import tensorflow as tf
from pyscf import dft, gto


class JAXDataset:
    def __init__(self, molecules):
        self.molecules = molecules
        self.data = [self._generate_features(mol) for mol in molecules]

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, index):
        return self.data[index]

    def _generate_features(self, molecule):
        mol = gto.M(atom=molecule)
        mean_field = dft.UKS(mol)
        return mean_field


# Example list of molecules
molecules = [
    [
        ["N", (0.0, 1.36627479, -0.21221668)],
        ["N", (0.0, -1.36627479, -0.21221668)],
        ["H", (-0.84470931, 1.89558816, 1.42901383)],
        ["H", (0.84470931, -1.89558816, 1.42901383)],
        ["H", (1.8260461, 1.89558816, 0.05688087)],
        ["H", (-1.8260461, -1.89558816, 0.05688087)],
    ]
]

dataset = JAXDataset(molecules)


def generator():
    for i in range(len(dataset)):
        yield dataset[i]


tf_dataset = tf.data.Dataset.from_generator(
    generator, output_signature=tf.TensorSpec(shape=(), dtype=tf.float32)
)
dataloader = tf_dataset.batch(2)

for batch in dataloader:
    print("Batch:", batch)
