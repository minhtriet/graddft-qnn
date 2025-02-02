import dataclasses

import flax.linen as nn
import numpy as np
import pcax
import pennylane as qml
from flax.typing import Array
from jaxlib.xla_extension import ArrayImpl

from graddft_qnn.standard_scaler import StandardScaler

from graddft_qnn.gates.ansatz import Ansatz
from graddft_qnn.helper.operatize import Operatize
from scipy.linalg import expm

from graddft_qnn.unitary_rep import O_h


@dataclasses.dataclass
class DFTQNN(nn.Module):
    """
    Here we define the circuit as well as the gates
    """

    dev: qml.device

    @nn.compact
    def __call__(self, feature: Array) -> Array:
        @qml.qnode(self.dev)
        def circuit(feature, psi, theta, phi, equivar_gate_matrix):
            """
            :param instance: an instance of the class Functional.
            :param rhoinputs: input to the neural network, in the form of an array.
            :return:
            """
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            for i in self.dev.wires[::3]:
                qml.QubitUnitary(equivar_gate_matrix, wires=range(i, i + 3))

            return qml.probs()

        # will calculate the coeff input without any dim reduction, might need to change that later.
        # feature = self.dim_reduction(feature)
        psi = self.param("psi", nn.initializers.normal(), (len(self.dev.wires),))
        theta = self.param("theta", nn.initializers.normal(), (len(self.dev.wires),))
        phi = self.param("phi", nn.initializers.normal(), (len(self.dev.wires),))

        unitary_reps = [O_h._180_deg_rot()]
        ansatz = Ansatz(np.pi, np.pi, np.pi, np.pi, [0, 1, 2])
        generator = DFTQNN.twirling(
            unitary_reps=unitary_reps, ansatz=qml.matrix(ansatz)
        )

        # this won't work
        # generator_op = Operatize(generator)
        # equivar_gate_matrix = qml.matrix(qml.evolve(generator_op))

        equivar_gate_matrix = expm(generator)

        result = circuit(feature, psi, theta, phi, equivar_gate_matrix)
        return result

    @staticmethod
    def twirling(ansatz, unitary_reps):
        generator = np.zeros_like(ansatz)
        for unitary_rep in unitary_reps:
            generator += unitary_rep @ ansatz @ unitary_rep.conjugate()
        generator /= len(ansatz)
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
