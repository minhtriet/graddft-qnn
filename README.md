## Setup
- Install poetry (`pipx install poetry`)
- `poetry install --with data_exploration`
## How to run an experiment
- Edit the config in `config.yaml`
- `python grad_dft/main.py`
- The run results can be seen in `reports.json`
## Current investigations
Te The training loss is the same, but the 
Try to run on bigger number of qubits (maximum 9 qubits)


## Dataset
1. (in use) Different bond lengths of H2 in training set and test set
2. Different molecules with default Pennlylane bond length

# Contribution

- A way to automatically generate ansatz given a group, regardless of the dimension or number of qubits
  - Both in matrix form and quantum gates form
- Invariant quantum neural network applied to DFT
- Performance comparison with normal n
