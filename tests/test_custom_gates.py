import numpy as np
import pytest

from graddft_qnn import custom_gates


@pytest.mark.skip(
    "We are not using that many gates, now it's just unitaries magic from pennylane"
)
def test_generate_operators():
    theta = 3.14
    matrix = custom_gates.generate_R_pauli(theta, ["Z", "Z", "Z"])
    assert np.allclose(matrix, custom_gates.RZZZ(theta))
    matrix = custom_gates.generate_R_pauli(theta, ["Y", "Z", "Y"])
    assert np.allclose(matrix, custom_gates.RYZY(theta))
