import functools

import numpy as np
import pennylane as qml
import scipy
from jax.scipy.linalg import expm


def U1(theta, i):
    qml.RX(theta[0], i)
    qml.RX(theta[1], i + 1)
    qml.RX(theta[2], i + 2)
    qml.MultiRZ(theta[3], [i, i + 1, i + 2])


def U2(theta, i):
    """
    The 16 invariant Pauli words with C2 are
    - XXX, XXI, XIX, IXX, XII, IXI, IIX
    - (Y, Z) x (YZ, ZY, YY, ZZ)
    :param theta:
    :param i:
    :return:
    """
    qml.QubitUnitary(RXXX(theta[0]), [i, i + 1, i + 2])
    qml.QubitUnitary(RXX(theta[1]), [i, i + 1])  # XXI
    qml.QubitUnitary(RXIX(theta[2]), [i, i + 1, i + 2])
    qml.RX(theta[3], i)
    qml.QubitUnitary(RYYY(theta[4]), [i, i + 1, i + 2])
    qml.QubitUnitary(RYYZ(theta[5]), [i, i + 1, i + 2])
    qml.QubitUnitary(RYZY(theta[6]), [i, i + 1, i + 2])
    qml.QubitUnitary(RYZZ(theta[7]), [i, i + 1, i + 2])
    qml.QubitUnitary(RZYY(theta[8]), [i, i + 1, i + 2])
    qml.QubitUnitary(RZYZ(theta[9]), [i, i + 1, i + 2])
    qml.MultiRZ(theta[10], [i, i + 1, i + 2])
    qml.QubitUnitary(RZZY(theta[11]), [i, i + 1, i + 2])
    qml.QubitUnitary(RXX(theta[12]), [i + 1, i + 2])  # IXX
    qml.RX(theta[13], i + 1)
    qml.RX(theta[14], i + 2)


def U2_measurement(i):
    return [
        qml.expval(qml.X(i) @ qml.X(i + 1) @ qml.X(i + 2)),
        qml.expval(qml.X(i) @ qml.X(i + 1)),
        qml.expval(qml.X(i) @ qml.I(i + 1) @ qml.X(i + 2)),
        qml.expval(qml.X(i)),
        qml.expval(qml.Y(i) @ qml.Y(i + 1) @ qml.Y(i + 2)),
        qml.expval(qml.Y(i) @ qml.Y(i + 1) @ qml.Z(i + 2)),
        qml.expval(qml.Y(i) @ qml.Z(i + 1) @ qml.Y(i + 2)),
        qml.expval(qml.Y(i) @ qml.Z(i + 1) @ qml.Z(i + 2)),
        qml.expval(qml.Z(i) @ qml.Y(i + 1) @ qml.Y(i + 2)),
        qml.expval(qml.Z(i) @ qml.Y(i + 1) @ qml.Z(i + 2)),
        qml.expval(qml.Z(i) @ qml.Z(i + 1) @ qml.Z(i + 2)),
        qml.expval(qml.Z(i) @ qml.Z(i + 1) @ qml.Y(i + 2)),
        qml.expval(qml.X(i + 1) @ qml.X(i + 2)),
        qml.expval(qml.X(i + 1)),
        qml.expval(qml.X(i + 2)),
        qml.expval(qml.I(i) @ qml.I(i + 1) @ qml.I(i + 2)),
    ]


def U2_6_wires(theta, i):
    raise NotImplementedError


def U2_6_wires_measurement(i):
    raise NotImplementedError


def RXXX(theta=3.1415):
    cos = np.cos(theta * 0.5)
    sin = -1j * np.sin(theta * 0.5)
    return np.array(
        [
            [cos, 0, 0, 0, 0, 0, 0, sin],
            [0, cos, 0, 0, 0, 0, sin, 0],
            [0, 0, cos, 0, 0, sin, 0, 0],
            [0, 0, 0, cos, sin, 0, 0, 0],
            [0, 0, 0, sin, cos, 0, 0, 0],
            [0, 0, sin, 0, 0, cos, 0, 0],
            [0, sin, 0, 0, 0, 0, cos, 0],
            [sin, 0, 0, 0, 0, 0, 0, cos],
        ]
    )


def RXIX(theta=1):
    return expm([-0.5j * theta * qml.matrix(qml.X(0) @ qml.I(1) @ qml.X(2))])


def RXX(theta=1):
    return expm([-0.5j * theta * qml.matrix(qml.X(0) @ qml.X(1))])


def RYYY(theta=3.1415):
    return expm([-0.5j * theta * qml.matrix(qml.Y(0) @ qml.Y(1) @ qml.Y(2))])


def RYYZ(theta=3.1415):
    return expm([-0.5j * theta * qml.matrix(qml.Y(0) @ qml.Y(1) @ qml.Z(2))])


def RYZY(theta=3.1415):
    return scipy.linalg.expm(
        [-0.5j * theta * qml.matrix(qml.Y(0) @ qml.Z(1) @ qml.Y(2))]
    )


def RYZZ(theta=3.1415):
    return expm([-0.5j * theta * qml.matrix(qml.Y(0) @ qml.Z(1) @ qml.Z(2))])


def RZYY(theta=3.1415):
    return expm([-0.5j * theta * qml.matrix(qml.Z(0) @ qml.Y(1) @ qml.Y(2))])


def RZYZ(theta=3.1415):
    return expm([-0.5j * theta * qml.matrix(qml.Z(0) @ qml.Y(1) @ qml.Z(2))])


def RZZZ(theta=3.1415):
    return scipy.linalg.expm(
        [-0.5j * theta * qml.matrix(qml.Z(0) @ qml.Z(1) @ qml.Z(2))]
    )


def RZZY(theta=3.1415):
    return expm([-0.5j * theta * qml.matrix(qml.Z(0) @ qml.Z(1) @ qml.Y(2))])


def generate_R_pauli(theta: float, pauli_string: list | str) -> qml.ops.op_math.Exp:
    """
    Takes in a p={X,Y,Z,I}^n string and output Rp
    For example: generate_R_pauli(pi, ["X", "Y", "Z", "X"] generates a R_XYZX gate
    """
    if isinstance(pauli_string, str):
        pauli_string = list(pauli_string)
    assert set(pauli_string).issubset({"X", "Y", "Z", "I"})
    ops_prod = qml.prod(
        *[getattr(qml, word)(idx) for idx, word in enumerate(pauli_string)]
    )
    return qml.exp(-0.5j * theta * ops_prod)


def generate_operators(pauli_string: str) -> qml.ops.op_math.Prod:
    if isinstance(pauli_string, str):
        pauli_string = list(pauli_string)
    assert set(pauli_string).issubset({"X", "Y", "Z", "I"})
    ops = [getattr(qml, word) for word in pauli_string]
    operators = [op(wires=i) for i, op in enumerate(ops)]
    combined_op = functools.reduce(lambda op1, op2: op1 @ op2, operators)
    return combined_op


words = {
    "X": qml.X.compute_matrix(),
    "Y": qml.Y.compute_matrix(),
    "Z": qml.Z.compute_matrix(),
    "I": qml.I.compute_matrix(),
}
