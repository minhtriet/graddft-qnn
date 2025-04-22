import logging

import pennylane as qml
import tqdm

from datasets import Dataset, DatasetDict, Features, Sequence, Value


class H2MultibondDataset:
    """
    The dataset contains quantum chemistry data for a fixed list of molecules
    """

    # List of molecule names as a class attribute
    mol_name = "H2"

    # Class attribute to cache the dataset
    _dataset = None

    @classmethod
    def get_dataset(cls):
        """
        Retrieves the DatasetDict containing train and test datasets.
        The dataset is built only once and cached for subsequent calls.

        Returns:
            DatasetDict: A dictionary with 'train' and 'test' datasets.
        """
        if cls._dataset is None:
            cls._dataset = cls.build_dataset()
        return cls._dataset

    @classmethod
    def build_dataset(cls):
        """
        Builds the DatasetDict by splitting molecules into train/test sets,
        generating data, and defining dataset features.

        Returns:
            DatasetDict: A dictionary with 'train' and 'test' datasets.
        """
        logging.info("Start building dataset")
        data_entries = qml.data.load(
            "qchem", molname=cls.mol_name, basis="STO-3G", bondlength="full"
        )
        train_size = int(0.7 * len(data_entries))
        train_mols = data_entries[:train_size]
        test_mols = data_entries[train_size:]

        # Generate data for train and test sets
        train_data = cls.generate_data(train_mols)
        test_data = cls.generate_data(test_mols)

        features = Features(
            {
                "name": Value("string"),  # Molecule name
                "groundtruth": Value("float64"),  # FCI energy
                "symbols": Sequence(Value("string")),  # List of atomic symbols
                "coordinates": Sequence(Sequence(Value("float64"), length=3)),
            }
        )

        train_dataset = Dataset.from_list(train_data, features=features)
        test_dataset = Dataset.from_list(test_data, features=features)
        return DatasetDict({"train": train_dataset, "test": test_dataset})

    @classmethod
    def generate_data(cls, data_entries):
        """
        Generates data for a list of molecules by loading quantum chemistry data.

        Args:
            mol_list (list): List of molecule names to process.

        Returns:
            list: List of dictionaries containing molecule data.
        """
        data = []
        for data_entry in tqdm.tqdm(data_entries):
            symbols = data_entry.molecule.symbols
            coordinates = data_entry.molecule.coordinates.astype(float)
            fci_e = data_entry.fci_energy
            example = {
                "name": cls.mol_name,
                "groundtruth": fci_e,
                "symbols": symbols,
                "coordinates": coordinates,
            }
            data.append(example)
        return data
