import dataclasses
import logging

import flax.linen as nn
import numpy as np
import pcax
import pennylane as qml
from flax.typing import Array
from jaxlib.xla_extension import ArrayImpl

from graddft_qnn import custom_gates
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
            :return: should be 1 measurement, so that graddft_qnn.qnn_functional.QNNFunctional.xc_energy works
            """
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            # 1st layer
            for i in self.dev.wires[::3]:
                custom_gates.U1(theta, i)
            # make sure the measurement is unique for each point
            return (
                qml.expval(qml.X(0)),
                # qml.expval(qml.X(1)),
                # qml.expval(qml.X(2)),
                # qml.expval(qml.Z(0) @ qml.Z(1) @ qml.Z(2)),
            )

        theta = self.param("theta", nn.initializers.normal(), (4,))

        result = circuit(feature, theta)
        return result

    @staticmethod
    def twirling(ansatz: np.array, unitary_reps: list[np.array]):
        generator = np.zeros_like(ansatz, dtype=np.complex64)
        ansatz = ansatz.astype(np.complex64)
        for unitary_rep in unitary_reps:
            generator += unitary_rep @ ansatz @ unitary_rep.conjugate()
        # 0.5 * (ansatz + unitary_rep @ ansatz @ unitary_rep.conjugate()
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
