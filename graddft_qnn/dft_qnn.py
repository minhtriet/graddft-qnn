import dataclasses
import logging
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pennylane as qml
from flax.typing import Array

from graddft_qnn import custom_gates


@dataclasses.dataclass
class DFTQNN(nn.Module):

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
            custom_gates.U2_6_wires(theta, 0)
            return custom_gates.U2_6_wires_measurement(0)

        jax.config.update("jax_enable_x64", True)
        theta = self.param('theta', nn.initializers.he_normal(), (2**len(self.dev.wires),1), jnp.float32)
        result = circuit(feature, theta)
        # result shape should be (grid*grid*grid, 1)
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
