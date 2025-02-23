import numpy as np
import pennylane as qml
from scipy.linalg import expm


def U1(theta, i):
    qml.RX(theta[0], i)
    qml.RX(theta[1], i + 1)
    qml.RX(theta[2], i + 2)
    qml.MultiRZ(theta[3], [i, i + 1, i + 2])


def RXX(theta=1):
    return expm(-1j * (0.5 * theta) * qml.matrix(qml.X(0) @ qml.X(1)))


def RXXX_matrix(theta=3.1415):
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
