## Setup
- Install poetry (`pipx install poetry`)
- Run `poetry install --with data_explore`
## How to run an experiment
- Edit the config in `config.yaml`
- `python graddft_qnn/main.py`
- The current and past run results can be seen in `reports.json`
  - To assist merging different reports, we have `consolidate_report.py`   
- Plot the train/test losses: `train_test_qubits_plot.py`
## Current investigations
- The training loss is the same
![Figure_1](https://github.com/user-attachments/assets/293b1e7f-c87e-4195-9fe9-c2568f316ec6)
but the test is different
![Figure_1](https://github.com/user-attachments/assets/a23bd026-8860-4b59-b17a-9b3ac6a7a92e)
- Try to run on bigger number of qubits (maximum 9 qubits)

## Datasets
1. (in use) Different bond lengths of H2 in training set and test set
2. H2 + Li2 + LiH
3. Different molecules with default Pennlylane bond length

# Contribution

- A way to automatically generate ansatz given a group, regardless of the dimension or number of qubits
  - Both in matrix form and quantum gates form
- Invariant quantum neural network applied to DFT
- Performance comparison with normal n
