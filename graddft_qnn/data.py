import logging
import os.path
import tempfile

import numpy as np
import requests
from pyscf import gto, scf
from pyscf.tools import cubegen
from rdkit import Chem


class DataGenerator:
    """
    create .cube files for different molecules
    """

    cids = {
        "h2o": "962",
        "h2": "783",
        "o2": "977",
    }
    REQUEST_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{0}/record/SDF?record_type=3d"

    def __init__(self):
        self.temp_file = tempfile.TemporaryDirectory()

    def __call__(self, *args, **kwargs):
        # todo could be written in aiohttp and asyncio later
        # todo tqdm
        for molecule, cid in DataGenerator.cids.items():
            res = requests.get(DataGenerator.REQUEST_URL.format(cid))
            with open(os.path.join(self.temp_file.name, f"{cid}.sdf"), "w") as file:
                file.write(res.text)

            suppl = Chem.SDMolSupplier(file.name)
            mol = next((m for m in suppl if m is not None), None)
            if not mol:
                logging.warning(f"Invalid scf file {molecule}")
                continue

            atoms = []
            coords = []

            for atom in mol.GetAtoms():
                atoms.append(atom.GetSymbol())

            # Extract 3D coordinates of the molecule from the conformer
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                coords.append(conf.GetAtomPosition(i))

            coords = np.array(coords)

            # Step 4: Convert the molecule to a PySCF Mol object
            # Create a list of atoms and their coordinates for PySCF
            atom_list = []
            for i, symbol in enumerate(atoms):
                atom_list.append((symbol, tuple(coords[i])))

            # Create a PySCF molecule object
            mol = gto.M(atom=atom_list, basis="6-31g*")

            mf = scf.RHF(mol).run()
            cubegen.density(mol, f"{molecule}_den.cube", mf.make_rdm1())


class DataLoader:
    """
    split data into train val and test set
    """

    pass


DataGenerator()()
