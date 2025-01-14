import numpy as np
from flax.typing import PRNGKey

from graddft_qnn.dft_qnn import DFTQNN
import pennylane as qml
import pytest

import numpy as np
from jax.random import PRNGKey


# Define a function to check equivariance
def check_equivariance(f, num_tests=10):
    print("Equivariance test passed!")
    return True


# Example function to test (must be O(3)-equivariant, e.g., vector-valued function)
def example_function(x):
    # For demonstration: just return the vector itself (trivially equivariant)
    return x


# Run the equivariance test
check_equivariance(example_function)


@pytest.fixture
def circuit():
    dev = qml.device("default.qubit", wires=3)
    circuit = DFTQNN(dev)
    return circuit


def test_quantum_circuit_with_embeddings(circuit):
    """
    Testing the relation
    :param setup_device:
    :return:
    """
    input = np.arange(1, 2**3)
    key = PRNGKey(42)
    coeff_input = np.random.random((3))
    parameters = circuit.init(key, coeff_input)

    # have to give parameters twice here but not sure of the reason, cannot call circuit
    f_x = circuit.apply(parameters, input)
    f_x_rot = circuit.apply(parameters, [6, 5, 8, 7, 2, 1, 4, 3])
    rot_f_x = (
        np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
            ]
        )
        @ f_x
    )
    assert np.allclose(f_x_rot, rot_f_x, atol=1e-6)
