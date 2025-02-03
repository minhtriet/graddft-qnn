from typing import Any

import pennylane as qml
import pennylane.numpy as np


class Ansatz(qml.operation.Operation):
    """
    XYZ
    XYZ --- ZZZ
    XYZ
    equation 55 JJMeyer
    """
    num_wires = 3
    num_params = 4

    def __init__(self, theta_x, theta_y, theta_z, theta_entanglement, wires):
        """
        thetas are placeholder in case we want them to be learnable params
        """
        super().__init__(theta_x, theta_y, theta_z, theta_entanglement, wires=wires)
        self._parameters = [theta_x, theta_y, theta_z, theta_entanglement]

    @staticmethod
    def compute_matrix(*params, **hyperparams: dict[str, Any]):
        per_wire = (
            qml.X.compute_matrix() @ qml.Y.compute_matrix() @ qml.Z.compute_matrix()
        )
        all_wires = np.kron(np.kron(per_wire, per_wire), per_wire)
        all_wires = all_wires @ Ansatz._ZZZ_matrix()

        return all_wires

    @staticmethod
    def _RXXX_matrix(theta=3.1415):
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

    def _ZZZ_matrix(self):
        return np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1]]
        )