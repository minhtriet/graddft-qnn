import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from flax.typing import Array


class DFTQNN(nn.Module):
    dev: qml.device
    ansatz_gen: list[np.array]
    gate_indices: list[int]
    rotate_matrix: np.array = None  # add this to test equivar
    rotate_feature: bool = False
    # rotate_feature: if True and rotate_matrix, then apply rotation to input feature,
    # otherwise apply rotation to output

    def setup(self) -> None:
        def _circuit(feature, theta, gate_gens):
            """
            :return: should be full measurement or just 1 measurement,
            so that graddft_qnn.qnn_functional.QNNFunctional.xc_energy works
            """
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            for idx, gen in enumerate(gate_gens):
                qml.exp(-1j * theta[idx][0] * gen)
            return qml.probs(wires=self.dev.wires)

        def _tn(feature, theta, gate_gens):
            """
            :return: should be full measurement or just 1 measurement,
            so that graddft_qnn.qnn_functional.QNNFunctional.xc_energy works
            """
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            for idx, gen in enumerate(gate_gens):
                qml.TrotterProduct(
                    -1j * theta[idx][0] * gen, time=2, order=2, check_hermitian=False
                )
            # tn doesn't support probs measurement like default.qubits
            # https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.default_tensor.DefaultTensor.html
            return qml.state()

        qml.decomposition.enable_graph()
        if self.dev.name == "default.qubit":
            self.qnode = qml.QNode(_circuit, self.dev)
            self.qnode = jax.jit(self.qnode)
        elif self.dev.name == "default.tensor":
            self.qnode = qml.QNode(_tn, self.dev)
        else:
            raise ValueError("unsupported qubit type")

    def circuit(self, feature, theta, gate_gens):
        if self.rotate_matrix is not None and self.rotate_feature:
            feature = self.rotate_matrix @ feature
        result = self.qnode(feature, theta, gate_gens)
        if self.rotate_matrix is not None and (not self.rotate_feature):
            result = self.rotate_matrix @ result
        if self.dev.name == "default.tensor":
            # convert state to prob
            result = jnp.abs(result) ** 2
            result /= jnp.sum(result)
            assert np.isclose(sum(result), 1)
        return jnp.array(result)

    @nn.compact
    def __call__(self, feature: Array) -> Array:
        theta = self.param(
            "theta",
            nn.initializers.he_normal(),
            (len(self.gate_indices), 1),
            jnp.float32,
        )
        selected_gates_gen = list(map(lambda i: self.ansatz_gen[i], self.gate_indices))
        return self.circuit(
            feature,
            theta,
            selected_gates_gen,
        )
