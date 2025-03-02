import dataclasses
import logging

import flax.linen as nn
import numpy as np
import pennylane as qml
from flax.typing import Array

from graddft_qnn import custom_gates


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
            # todo find all invariant Pauli words (2^9N tensor product) (only needs N=9 measurements)
            # todo the test should return a vector rather than a scalar?
            # todo run the network of the gradDFT to see the shapes of output

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
    def twirling(ansatz: np.array, unitary_reps: list[np.array], print_debug=False):
        ansatz = ansatz.astype(np.complex64)
        generator = np.array(ansatz)  # deep copy
        for unitary_rep in unitary_reps:
            generator += unitary_rep @ ansatz @ unitary_rep.conjugate()
        generator /= len(unitary_reps) + 1
        if np.allclose(generator, np.zeros_like(generator)):
            if print_debug:
                logging.info("This ansatz gate doesn't work with this group")
            return None
        return generator

    @staticmethod
    def twirling_(ansatz: np.array, unitary_reps: list[np.array]):
        generator = np.zeros_like(ansatz, dtype=np.complex64)
        ansatz = ansatz.astype(np.complex64)
        for unitary_rep in unitary_reps:
            twirled = 0.5 * (ansatz + unitary_rep @ ansatz @ unitary_rep.conjugate())
            if np.allclose(twirled, np.zeros_like(twirled)):
                print("All zero")
            else:
                print(twirled)
                print(qml.pauli_decompose(twirled))
            print()
        return twirled
