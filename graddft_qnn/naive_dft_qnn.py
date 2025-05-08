import logging
import math
from itertools import combinations

import flax.linen as nn
import jax.numpy as jnp
import pennylane as qml
from flax.typing import Array


class NaiveDFTQNN(nn.Module):
    dev: qml.device
    measurements: list[qml.operation.Operator]

    def _circuit(self, feature, theta):
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
        return [qml.expval(z_op) for z_op in self.measurements]

    def setup(self) -> None:
        self.qnode = qml.QNode(self._circuit, self.dev)

    def circuit(self, feature, theta):
        return jnp.array(self.qnode(feature, theta))

    @nn.compact
    def __call__(self, feature: Array) -> Array:
        theta = self.param(
            "theta",
            nn.initializers.he_normal(),
            (math.ceil(len(self.dev.wires) / 2), 1),
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

    @staticmethod
    def generate_Z_measurements(n):
        """
        Generate 2^n_wires Z string
        """
        logging.info("Generating Z measurements")
        combos = NaiveDFTQNN._combination(n)
        result = []
        for combo in combos:
            term = qml.Z(combo[0])
            for i in combo[1:]:
                term = term @ qml.Z(i)
            result.append(term)
        # add last measurement
        result.append(qml.Z(0))
        return result
