import numpy as np

from graddft_qnn import custom_gates


def test_generate_operators():
    theta = 3.14
    matrix = custom_gates.generate_R_pauli(theta, ["Z", "Z", "Z"])
    assert np.allclose(matrix, custom_gates.RZZZ(theta))
    matrix = custom_gates.generate_R_pauli(theta, ["Y", "Z", "Y"])
    assert np.allclose(matrix, custom_gates.RYZY(theta))
