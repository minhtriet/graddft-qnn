import pennylane as qml
import numpy as np

def RXX(theta):
    return np.cos(theta * 0.5) * np.eye(4) - 1j * np.sin(theta * 0.5) * np.kron(
        qml.X.compute_matrix(), qml.X.compute_matrix()
    )

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

def ZZZ_matrix():
    return np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1],
        ]
    )
