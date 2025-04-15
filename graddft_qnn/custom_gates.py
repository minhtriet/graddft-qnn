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


def U2_6_wires(theta, i):  # noqa: PLR0915
    """
    The 16 invariant Pauli words with C2 are
    - XXX, XXI, XIX, IXX, XII, IXI, IIX
    - (Y, Z) x (YZ, ZY, YY, ZZ)
    :param theta:
    :param i:
    :return:
    """
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXXXXX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[1], "XXXXXI"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[2], "XXXXYY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[3], "XXXXYZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[3], "XXXXZY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[4], "XXXXZZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[5], "XXXXIX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[6], "XXXXII"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[7], "XXXIXX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[8], "XXXIXI"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[9], "XXXIYY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[10], "XXXIYZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[11], "XXXIZY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[12], "XXXIZZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[13], "XXXIIX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[14], "XXXIII"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[14], "XXYYXX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[14], "XXYYXI"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[14], "XXYYYY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[14], "XXYYYZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYYZY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYYZZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYYIX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYYII"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYZXX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYZXI"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYZYY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYZYZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYZZY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYZZZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYZIX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXYZII"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZYXX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZYXI"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZYYY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZYYZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZYZY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZYZZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZYIX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZYII"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZZXX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZZXI"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZZYY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZZYZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZZZY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZZZZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZZIX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXZZII"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIXXX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIXXI"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIXYY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIXYZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIXZY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIXZZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIXIX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIXII"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIIXX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIIXI"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIIYY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIIYZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIIZY"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIIZZ"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIIIX"), range(i, i + 6))
    qml.QubitUnitary(generate_R_pauli(theta[0], "XXIIII"), range(i, i + 6))


def U2_6_wires_measurement(i):
    return (
        qml.expval(generate_operators("XXXXXX")),
        qml.expval(generate_operators("XXXXXI")),
        qml.expval(generate_operators("XXXXYY")),
        qml.expval(generate_operators("XXXXYZ")),
        qml.expval(generate_operators("XXXXZY")),
        qml.expval(generate_operators("XXXXZZ")),
        qml.expval(generate_operators("XXXXIX")),
        qml.expval(generate_operators("XXXXII")),
        qml.expval(generate_operators("XXXIXX")),
        qml.expval(generate_operators("XXXIXI")),
        qml.expval(generate_operators("XXXIYY")),
        qml.expval(generate_operators("XXXIYZ")),
        qml.expval(generate_operators("XXXIZY")),
        qml.expval(generate_operators("XXXIZZ")),
        qml.expval(generate_operators("XXXIIX")),
        qml.expval(generate_operators("XXXIII")),
        qml.expval(generate_operators("XXYYXX")),
        qml.expval(generate_operators("XXYYXI")),
        qml.expval(generate_operators("XXYYYY")),
        qml.expval(generate_operators("XXYYYZ")),
        qml.expval(generate_operators("XXYYZY")),
        qml.expval(generate_operators("XXYYZZ")),
        qml.expval(generate_operators("XXYYIX")),
        qml.expval(generate_operators("XXYYII")),
        qml.expval(generate_operators("XXYZXX")),
        qml.expval(generate_operators("XXYZXI")),
        qml.expval(generate_operators("XXYZYY")),
        qml.expval(generate_operators("XXYZYZ")),
        qml.expval(generate_operators("XXYZZY")),
        qml.expval(generate_operators("XXYZZZ")),
        qml.expval(generate_operators("XXYZIX")),
        qml.expval(generate_operators("XXYZII")),
        qml.expval(generate_operators("XXZYXX")),
        qml.expval(generate_operators("XXZYXI")),
        qml.expval(generate_operators("XXZYYY")),
        qml.expval(generate_operators("XXZYYZ")),
        qml.expval(generate_operators("XXZYZY")),
        qml.expval(generate_operators("XXZYZZ")),
        qml.expval(generate_operators("XXZYIX")),
        qml.expval(generate_operators("XXZYII")),
        qml.expval(generate_operators("XXZZXX")),
        qml.expval(generate_operators("XXZZXI")),
        qml.expval(generate_operators("XXZZYY")),
        qml.expval(generate_operators("XXZZYZ")),
        qml.expval(generate_operators("XXZZZY")),
        qml.expval(generate_operators("XXZZZZ")),
        qml.expval(generate_operators("XXZZIX")),
        qml.expval(generate_operators("XXZZII")),
        qml.expval(generate_operators("XXIXXX")),
        qml.expval(generate_operators("XXIXXI")),
        qml.expval(generate_operators("XXIXYY")),
        qml.expval(generate_operators("XXIXYZ")),
        qml.expval(generate_operators("XXIXZY")),
        qml.expval(generate_operators("XXIXZZ")),
        qml.expval(generate_operators("XXIXIX")),
        qml.expval(generate_operators("XXIXII")),
        qml.expval(generate_operators("XXIIXX")),
        qml.expval(generate_operators("XXIIXI")),
        qml.expval(generate_operators("XXIIYY")),
        qml.expval(generate_operators("XXIIYZ")),
        qml.expval(generate_operators("XXIIZY")),
        qml.expval(generate_operators("XXIIZZ")),
        qml.expval(generate_operators("XXIIIX")),
        qml.expval(generate_operators("XXIIII")),
    )


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
