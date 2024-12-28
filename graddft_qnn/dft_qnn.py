import dataclasses

import pcax
import pennylane as qml

from standard_scaler import StandardScaler
from jaxlib.xla_extension import ArrayImpl
import flax.linen as nn
from flax.typing import Array

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

            for i in self.dev.wires:
                qml.RY(theta[i], wires=i)
                qml.U1(phi[i], wires=i)
            return qml.probs()

        # todo I don't like this, but have to do because grad_dft.functional.Functional.compute_coefficient_inputs
        # will calculate the coeff input without any dim reduction, might need to change that later.
        feature = self.dim_reduction(feature)
        theta = self.param('theta', nn.initializers.normal(), (len(self.dev.wires), ))
        phi = self.param("phi", nn.initializers.normal(), (len(self.dev.wires), ))
        result = circuit(feature, theta, phi)
        return result

    # todo scale everytime or save the scaler?
    def dim_reduction(self, original_array: ArrayImpl):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(original_array)
        state = pcax.fit(X_scaled, n_components=1)
        X_pca = pcax.transform(state, X_scaled)
        X_pca = X_pca.flatten()
        X_pca = X_pca[:len(self.dev.wires)]
        return X_pca
