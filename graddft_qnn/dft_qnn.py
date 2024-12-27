import yaml
import pennylane as qml

import flax.linen as nn
from flax.typing import Array


class DFTQNN(nn.Module):
    def __init__(self, yaml_file):
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
            if "QBITS" not in data:
                raise KeyError("YAML file must contain 'QBITS' key")
            self.num_qubits = data["QBITS"]
            self.dev = qml.device("default.qubit", wires=self.num_qubits)

    @nn.compact
    def __call__(self, feature: Array) -> Array:

        @qml.device(self.dev)
        def circuit(feature, theta, phi):
            """
            :param instance: an instance of the class Functional.
            :param rhoinputs: input to the neural network, in the form of an array.
            :return:
            """
            qml.AmplitudeEmbedding(feature, wires=range(self.num_qubits), pad_with=0.0)

            for i in range(self.num_qubits):
                qml.RY(theta[i], wires=i)
                qml.U1(phi[i], wires=i)
            return qml.probs()


        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        theta = self.param('theta', self.param_init)
        phi = self.param("phi", self.param_init)
        return circuit(feature, theta, phi)

