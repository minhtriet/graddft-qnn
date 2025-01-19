import tempfile

import requests
import rdkit

class DataGenerator:
    """
    create .cube files for different molecules
    """
    cids = {
        'h2o': '962'
    }
    REQUEST_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{0}/record/SDF?record_type=3d"
    def __init__(self):
        # Create a temporary named file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)

        # Print the file name (can be used later)
        print(f"Temporary file created: {self.temp_file.name}")

    @staticmethod
    def download_pubchem():
        for cid in DataGenerator.cids.values():
            res = requests.get(DataGenerator.REQUEST_URL.format(cid))
            with open("conformer1.txt", 'w') as file:
                file.write(res.text)


class DataLoader:
    """
    split data into train val and test set
    """
    pass