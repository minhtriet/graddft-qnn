import dataclasses
import numpy as np
import flax.linen as nn
import pcax
import pennylane as qml
from flax.typing import Array
from jaxlib.xla_extension import ArrayImpl
from graddft_qnn.standard_scaler import StandardScaler


@dataclasses.dataclass
class DFTQNN(nn.Module):
    dev: qml.device

    @nn.compact
    def __call__(self, feature: Array) -> Array:
        @qml.qnode(self.dev)
        def circuit(feature, psi, theta, phi):
            """
            :param instance: an instance of the class Functional.
            :param rhoinputs: input to the neural network, in the form of an array.
            :return:
            """
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            for i in self.dev.wires[::3]:
                self.U_O3(psi[i], theta[i], phi[i], wires=range(i, i + 3))

            return qml.probs()

        # todo I don't like this, but have to do because grad_dft.functional.Functional.compute_coefficient_inputs
        # will calculate the coeff input without any dim reduction, might need to change that later.
        # feature = self.dim_reduction(feature)
        psi = self.param("psi", nn.initializers.normal(), (len(self.dev.wires),))
        theta = self.param("theta", nn.initializers.normal(), (len(self.dev.wires),))
        phi = self.param("phi", nn.initializers.normal(), (len(self.dev.wires),))
        result = circuit(feature, psi, theta, phi)
        return result

    # todo save the scaler instead of scaling everytime like now
    def dim_reduction(self, original_array: ArrayImpl):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(original_array)
        state = pcax.fit(X_scaled, n_components=1)
        X_pca = pcax.transform(state, X_scaled)
        X_pca = X_pca.flatten()
        X_pca = X_pca[: len(self.dev.wires)]
        return X_pca

    def U_O3(self, psi, theta, phi, wires, gamma=0):
        # todo change gamma to a learnable param
        qml.RZ(psi, wires=wires[0])
        qml.RX(theta, wires=wires[0])
        qml.RZ(phi, wires=wires[0])

        qml.RZ(psi, wires=wires[1])
        qml.RX(theta, wires=wires[1])
        qml.RZ(phi, wires=wires[1])

        qml.RZ(psi, wires=wires[2])
        qml.RX(theta, wires=wires[2])
        qml.RZ(phi, wires=wires[2])

        qml.QubitUnitary(self._RXXX_matrix(gamma), wires=wires[0:3])

    def _RXXX_matrix(self, theta):
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

    def V_O3(self, psi, theta, phi):
        pass

    def U1_AE(self, thetas, wires):
        # circuit 1
        qml.Rx(thetas[0], wires=wires[0])
        qml.Rx(thetas[1], wires=wires[1])
        qml.QubitUnitary(self._RXX_matrix(thetas[2]), wires=wires[0, 1])

    def U3_AE(self, theta_1, theta_2, theta_3, theta_4, theta_5, wires):
        # circuit 4
        qml.Rot(theta_1, theta_2, theta_3, wires=wires[0])
        qml.RX(theta_4, wires=wires[1])
        qml.QubitUnitary(self._RXX_matrix(theta_5), wires=wires)

    def V1_AE(self, phi_1, phi_2, control_wire, wire):
        # circuit 6
        qml.CRX(phi_1, wires=[control_wire, wire])
        qml.X(control_wire)
        qml.CRX(phi_2, wires=[control_wire, wire])
        qml.X(control_wire)

    def V3_AE(self, phi_1, phi_2, control_wire, wire):
        # circuit 7
        qml.CRZ(phi_1, wires=[control_wire, wire])
        qml.X(control_wire)
        qml.CRY(phi_2, wires=[control_wire, wire])
        qml.X(control_wire)

    def _X_matrix(self):
        return np.array([[0, 1], [1, 0]])

    def _RXX_matrix(self, theta):
        return np.cos(theta * 0.5) * np.eye(4) - 1j * np.sin(theta * 0.5) * np.kron(
            self._X_matrix(), self._X_matrix()
        )
