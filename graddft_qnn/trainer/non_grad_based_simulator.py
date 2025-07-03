import logging
from dataclasses import dataclass

import grad_dft as gd
import pennylane as qml
import tqdm
from pyscf import dft, gto

from datasets import DatasetDict
from graddft_qnn.trainer.base_simulator import Simulator

logging.getLogger().setLevel(logging.INFO)


@dataclass
class NonGradBasedSimulator(Simulator):
    dev: qml.devices.device_api.Device
    functional: gd.Functional
    dataset: DatasetDict

    def simulate(self):
        predictor = gd.non_scf_predictor(self.functional)
        train_ds = self.dataset["train"]
        for train_example in tqdm.tqdm(train_ds):
            atom_coords = list(
                zip(train_example["symbols"], train_example["coordinates"])
            )
            mol = gto.M(atom=atom_coords, basis="def2-tzvp")
            mean_field = dft.UKS(mol)
            mean_field.kernel()
            molecule = gd.molecule_from_pyscf(mean_field)
            parameters = [1, 2, 3]
            atoms_out = predictor(parameters, molecule)
            E_predict = atoms_out.energy
            diff = E_predict - train_example["groundtruth"]
            # todo minimize diff through parameters
