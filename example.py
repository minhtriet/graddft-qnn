from graddft_qnn.dft_qnn import DFTQNN
import numpy as np
import pennylane as qml
import functools

from graddft_qnn.unitary_rep import O_h


def process(gate_matrix, u_reprs: list[np.array]):
    gen = DFTQNN.twirling(gate_matrix, unitary_reps=u_reprs)
    if isinstance(gen, np.ndarray):
        return qml.pauli_decompose(
            gen, check_hermitian=False, hide_identity=True, pauli=True
        )
    return None


YII = functools.reduce(np.kron, [qml.X.compute_matrix(), np.eye(2), np.eye(2)])
IYI = functools.reduce(np.kron, [np.eye(2), qml.X.compute_matrix(), np.eye(2)])
IIY = functools.reduce(np.kron, [np.eye(2), np.eye(2), qml.X.compute_matrix()])

unitary_reps = O_h._180_deg_rot()

process(YII, unitary_reps)
