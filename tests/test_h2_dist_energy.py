import pathlib
from datasets import DatasetDict
import numpy as np
from graddft_qnn.cube_dataset.h2_multibond import H2MultibondDataset
import matplotlib.pyplot as plt
from pyscf import dft, fci, gto, scf
from collections import namedtuple
Molecule = namedtuple("Molecule", ["symbols", "coordinates"])
DataEntry = namedtuple(
    "DataEntry", ["molecule", "fci_energy"]
)  # Instantiate the object

def test_h2_dist_energy():
    energy_dict = h2_dist_energy() # dataset
    energy_dict_test = h2_dist_energy_test() # user setting
    energy_dict_extra = calculate_extra_data()

    plt.plot([float(x) for x in energy_dict.keys()],
             energy_dict.values(),
             color="blue",
             label="training and test dataset",
             )

    plt.plot([float(x) for x in energy_dict_test.keys()],
             energy_dict_test.values(),
             color="red",
             label="generate test dataset",
             )

    plt.plot([float(x) for x in energy_dict_extra.keys()],
             energy_dict_extra.values(),
             color="green",
             label="generate extra dataset",
             )

    plt.xlabel("Distance between H atoms (Å)")
    plt.ylabel("Energy (Hartree)")
    plt.title("Potential Energy Curves")
    #plt.xlim([0.11, 4.0])
    plt.ylim([-1.5, 2.5])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def h2_dist_energy():
    if pathlib.Path("datasets/h2_dataset").exists():
        dataset = DatasetDict.load_from_disk(pathlib.Path("datasets/h2_dataset"))
    else:
        dataset = H2MultibondDataset.get_dataset()
        dataset.save_to_disk("datasets/h2_dataset")

    h2_dist_energy = {}
    for key in ["train", "test"]:
        for data in dataset[key]:
            dist = np.linalg.norm(
                np.array(data["coordinates"][0]) - np.array(data["coordinates"][1])
            )
            if dist in [0.944865, 1.0960434, 1.0204542]:
                continue
            h2_dist_energy[dist] = data["groundtruth"]

    return dict(sorted(h2_dist_energy.items()))

def h2_dist_energy_test():
    h2_dist_energy = {}
    for distance in np.arange(0.1, 4.0, 0.05):
        # Create molecule with the specified distance
        mol = gto.M(
            atom=[["H", (0, 0, 0)], ["H", (0, 0, distance)]],
            basis="def2-tzvp",
            unit="Angstrom",
        )
        mean_field = dft.UKS(mol)
        ground_truth_energy = mean_field.kernel()
        h2_dist_energy[np.linalg.norm(distance)] = ground_truth_energy

    return dict(sorted(h2_dist_energy.items()))

def calculate_extra_data():
    data_entries = []
    for bond_length in np.arange(0.1, 4.0, 0.05):
        # Define the H2 molecule with the given bond length
        mol = gto.Mole()
        mol.atom = f"""
        H 0 0 0
        H 0 0 {bond_length}
        """
        mol.basis = "def2-tzvp"  # Use a decent basis set, e.g., cc-pVDZ
        mol.unit = "Angstrom"
        mol.build()

        # Perform Hartree-Fock calculation
        mf = scf.RHF(mol)
        mf.kernel()

        # Perform FCI calculation
        cisolver = fci.FCI(mf)
        fci_energy = cisolver.kernel()[0]
        molecule = Molecule(
            symbols=["H", "H"],
            coordinates=np.array([[0, 0, 0], [0, 0, bond_length]]),
        )
        data_entry = DataEntry(molecule=molecule, fci_energy=fci_energy)
        data_entries.append(data_entry)

    h2_dist_energy = {}
    for data in data_entries:
        dist = np.linalg.norm(
            np.array(data.molecule.coordinates[0]) - np.array(data.molecule.coordinates[1])
        )
        h2_dist_energy[dist] = data.fci_energy

    return dict(sorted(h2_dist_energy.items()))