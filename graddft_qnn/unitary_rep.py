import pennylane as qml
import pennylane.numpy as np


class O_h:
    @staticmethod
    def _180_deg_rot():
        return np.kron(
            np.kron(qml.X.compute_matrix(), qml.X.compute_matrix()),
            qml.I.compute_matrix(),
        )

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
