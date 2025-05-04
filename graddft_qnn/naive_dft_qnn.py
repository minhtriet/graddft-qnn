import math

import flax.linen as nn
import jax.numpy as jnp
import pennylane as qml
from flax.typing import Array


class NaiveDFTQNN(nn.Module):
    dev: qml.device

    def circuit(self, feature, theta):
        @qml.qnode(self.dev)
        def _circuit(feature, theta):
            """
            :return: should be full measurement or just 1 measurement,
            so that graddft_qnn.qnn_functional.QNNFunctional.xc_energy works
            """
            start_wire = 0
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            if len(self.dev.wires) % 2 == 1:
                qml.RX(theta[0][0], wires=0)
                start_wire = 1
            for i in range(start_wire, len(self.dev.wires), 2):
                qml.SingleExcitation(theta[i // 2][0], wires=[i, i + 1])
            return qml.state()

        result = jnp.real(_circuit(feature, theta))
        return result

    @nn.compact
    def __call__(self, feature: Array) -> Array:
        theta = self.param(
            "theta",
            nn.initializers.he_normal(),
            (math.ceil(len(self.dev.wires) / 2), 1),
            jnp.float32,
        )
        return self.circuit(feature, theta)
