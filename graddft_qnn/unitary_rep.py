import numpy as np
import pennylane as qml


class O_h:
    @staticmethod
    def _180_deg_rot():
        return [np.kron(
            np.kron(qml.X.compute_matrix(), qml.X.compute_matrix()),
            qml.I.compute_matrix(),
        ), np.eye(8)]

    @staticmethod
    def rz():
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    @staticmethod
    def _180_deg_rot_matrix():
        return np.array(
            [
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
            ]
        )
