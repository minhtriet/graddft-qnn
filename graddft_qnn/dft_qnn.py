import dataclasses
import logging

import flax.linen as nn
import numpy as np
import pcax
import pennylane as qml
from flax.typing import Array
from jaxlib.xla_extension import ArrayImpl

from graddft_qnn.standard_scaler import StandardScaler


@dataclasses.dataclass
class DFTQNN(nn.Module):
    """
    Here we define the circuit as well as the gates
    """

    dev: qml.device

    @nn.compact
    def __call__(self, feature: Array) -> Array:
        @qml.qnode(self.dev)
        def circuit(feature, theta):
            """
            :param instance: an instance of the class Functional.
            :param rhoinputs: input to the neural network, in the form of an array.
            :return:
            """
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            for i in self.dev.wires[::3]:
                qml.RX(theta[0], i)
                qml.RX(theta[1], i + 1)
                qml.RX(theta[2], i + 2)
                qml.RZ(theta[3], i)
                qml.RZ(theta[4], i + 1)
                qml.RZ(theta[5], i + 2)
                return (qml.expval(qml.X(0) @ qml.Z(0)),)
                (qml.expval(qml.X(1) @ qml.Z(1)),)
                qml.expval(qml.X(2) @ qml.Z(2))

        theta = self.param("theta", nn.initializers.normal(), (6,))

        result = circuit(feature, theta)
        return result

    @staticmethod
    def twirling(ansatz: np.array, unitary_reps: list[np.array]):
        generator = np.zeros_like(ansatz, dtype=np.complex64)
        ansatz = ansatz.astype(np.complex64)
        for unitary_rep in unitary_reps:
            generator += unitary_rep @ ansatz @ unitary_rep.conjugate()
        generator /= len(unitary_reps)
        if np.allclose(generator, np.zeros_like(generator)):
            logging.info("This ansatz gate doesn't work with this group")
            return None
        return generator

    # =========
    # todo save the scaler instead of scaling everytime like now
    def dim_reduction(self, original_array: ArrayImpl):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(original_array)
        state = pcax.fit(X_scaled, n_components=1)
        X_pca = pcax.transform(state, X_scaled)
        X_pca = X_pca.flatten()
        X_pca = X_pca[: len(self.dev.wires)]
        return X_pca
