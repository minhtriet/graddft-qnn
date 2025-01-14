
@dataclasses.dataclass
class DFTQNN(nn.Module):
    dev: qml.device

    @nn.compact
    def __call__(self, feature: Array) -> Array:
        @qml.qnode(self.dev)
        def circuit(feature, theta, phi):
            """
            :param instance: an instance of the class Functional.
            :param rhoinputs: input to the neural network, in the form of an array.
            :return:
            """
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            theta_idx = 0
            for i in self.dev.wires[:,:,3]:
                self.U_O3(wires=[i, i+2], theta_1=theta[theta_idx: theta_idx + 2])

            return qml.probs()

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
        qml.Rz(psi, wires=wires[0])
        qml.Rx(theta, wires=wires[0])
        qml.Rz(phi, wires=wires[0])

        qml.Rz(psi, wires=wires[1])
        qml.Rx(theta, wires=wires[1])
        qml.Rz(phi, wires=wires[1])

        qml.Rz(psi, wires=wires[2])
        qml.Rx(theta, wires=wires[2])
        qml.Rz(phi, wires=wires[2])

        qml.QubitUnitary(self._RXXX_matrix(gamma), wires=wires[0, 1, 2])

    
    def _RXXX_matrix(self, theta):
        cos = np.cos(theta * 0.5)
        sin = -1j*np.sin(theta * 0.5)
        return np.array([
                [cos, 0, 0, 0, 0, 0, 0, sin],
                [0, cos, 0, 0, 0, 0, sin, 0],
                [0, 0, cos, 0, 0, sin, 0, 0],
                [0, 0, 0, cos, sin, 0, 0, 0],
                [0, 0, 0, sin, cos, 0, 0, 0],
                [0, 0, sin, 0, 0, cos, 0, 0],
                [0, sin, 0, 0, 0, 0, cos, 0],
                [sin, 0, 0, 0, 0, 0, 0, cos],
                ])
