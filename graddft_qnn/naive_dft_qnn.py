import logging
from itertools import combinations

import flax.linen as nn
import jax.numpy as jnp
import pennylane as qml
from flax.typing import Array


class NaiveDFTQNN(nn.Module):
    dev: qml.device
    num_gates: int

    def _circuit(self, feature, theta):
        """
        :return: should be full measurement or just 1 measurement,
        so that graddft_qnn.qnn_functional.QNNFunctional.xc_energy works
        """
        qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
        num_layer = (self.num_gates // (len(self.dev.wires) * 3)) + 1
        angles = []
        for i in range(len(self.dev.wires) * 3 * num_layer):
            if i < self.num_gates:
                angles.append(theta[i][0])
            else:
                angles.append(theta[self.num_gates][0])
            if (i + 1) % 3 == 0:
                qml.Rot(
                    angles[0],
                    angles[1],
                    angles[2],
                    wires=((i + 1) // 3 - 1) % len(self.dev.wires),
                )
                angles = []
            if (i + 1) % (len(self.dev.wires) * 3) == 0:
                for j in range(len(self.dev.wires) - 1):
                    qml.CNOT(wires=[j, j + 1])
        return qml.probs(wires=self.dev.wires)

    def setup(self) -> None:
        self.qnode = qml.QNode(self._circuit, self.dev)

    def circuit(self, feature, theta):
        return jnp.array(self.qnode(feature, theta))

    @nn.compact
    def __call__(self, feature: Array) -> Array:
        theta = self.param(
            "theta",
            nn.initializers.he_normal(),
            (self.num_gates, 1),
            jnp.float32,
        )
        return self.circuit(feature, theta)

    @staticmethod
    def _combination(n):
        r"""
        Return total number of non-empty combinations of a set with n element
        to use as measurements
        There are \sum_{k=1}^n (n k) = 2^n - 1
        Input: 3
        output: [0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]
        """
        result = []
        for r in range(1, n + 1):
            for combo in combinations(range(n), r):
                result.append(list(combo))
        return result