import numpy as np
import pennylane as qml
import pytest

from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.unitary_rep import O_h


def test_twirling():
    # eq.60 in https://doi.org/10.1103/PRXQuantum.4.010328
    twirled = DFTQNN._twirling(qml.X(0) @ qml.I(1), [qml.SWAP([0, 1])])
    expected = 0.5 * (qml.X(0) + qml.X(1))
    assert np.allclose(qml.matrix(twirled), qml.matrix(expected))

    # eq.65 in https://doi.org/10.1103/PRXQuantum.4.010328
    twirled = DFTQNN._twirling(qml.I(0) @ qml.Y(1), [qml.X(0) @ qml.X(1)])
    assert twirled is None


@pytest.mark.skip(reason="takes too long, can run separately")
def test_gate_design():
    num_wire = 6
    dev = qml.device("default.qubit", wires=num_wire)

    @qml.qnode(dev)
    def six_qubit_circuit(params, gate_gens):
        qml.AmplitudeEmbedding(params, wires=range(6), normalize=True)
        for gate_gen in gate_gens:
            qml.exp(-0.5j * gate_gen)
        return [qml.expval(gate_gen) for gate_gen in gate_gens]

    gates_gen_sparse = DFTQNN.gate_design(6, [O_h._180_deg_x_rot_sparse(4, True)])
    gates_gen_dense = DFTQNN.gate_design(6, [O_h._180_deg_x_rot(4, True)])
    np.random.seed(33)
    input = np.random.rand(2**6)
    output_dense = six_qubit_circuit(input, gates_gen_dense)
    output_sparse = six_qubit_circuit(input, gates_gen_sparse)
    assert np.allclose(output_dense, output_sparse)
