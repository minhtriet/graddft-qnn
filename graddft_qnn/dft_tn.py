from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pennylane as qml


@dataclass
class DFTTN:
    dev: qml.device
    ansatz_gen: list[np.array]
    gate_indices: list[int]

    def __post_init__(self) -> None:
        def _circuit(feature, theta, gate_gens):
            """
            :return: should be full measurement or just 1 measurement,
            so that graddft_qnn.qnn_functional.QNNFunctional.xc_energy works
            """
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            for idx, gen in enumerate(gate_gens):
                qml.TrotterProduct(
                    -1j * theta[idx] * gen, time=2, order=2, check_hermitian=False
                )
            # tn doesn't support probs measurement like default.qubits
            # https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.default_tensor.DefaultTensor.html
            return qml.state()

        qml.decomposition.enable_graph()
        self.theta = [1,2,3]
        self.qnode = qml.QNode(_circuit, self.dev, interface="numpy")

    def circuit(self, feature, theta, gate_gens):
        result = self.qnode(feature, theta, gate_gens)
        # convert state to prob
        result = jnp.abs(result) ** 2
        result /= jnp.sum(result)
        assert jnp.isclose(jnp.sum(result), 1)
        return jnp.array(result)

    def __call__(self, feature):
        selected_gates_gen = list(map(lambda i: self.ansatz_gen[i], self.gate_indices))
        return self.circuit(
            feature,
            self.theta,
            selected_gates_gen,
        )