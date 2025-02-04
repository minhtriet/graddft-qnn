import pennylane as qml
import pennylane.numpy as np


class MyGate(qml.operation.Operation):
    num_wires = 1
    num_params = 2

    def __init__(self, angle_x, angle_y, wires):
        super().__init__(angle_x, angle_y, wires=wires)
        qml.X
        self._parameters = [angle_x, angle_y]

    @staticmethod
    def compute_matrix(*parameters):
        theta_x, theta_y = parameters
        rx = np.array(
            [
                [np.cos(theta_x / 2), -1j * np.sin(theta_x / 2)],
                [-1j * np.sin(theta_x / 2), np.cos(theta_x / 2)],
            ]
        )
        ry = np.array(
            [
                [np.cos(theta_y / 2), -np.sin(theta_y / 2)],
                [np.sin(theta_y / 2), np.cos(theta_y / 2)],
            ]
        )
        return ry @ rx  # Multiplying matrices in reverse order for correct application


# print(qml.matrix(qml.evolve(qml.X(1))))
# print(qml.matrix(qml.evolve(MyGate(1,2,1))))
