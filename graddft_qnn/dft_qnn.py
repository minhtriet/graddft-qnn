import itertools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from flax.typing import Array
from tqdm import tqdm

from graddft_qnn import custom_gates
from graddft_qnn.unitary_rep import is_zero_matrix_combination


class DFTQNN(nn.Module):
    dev: qml.device
    ansatz_gen: list[np.array]
    gate_indices: list[int]

    def setup(self) -> None:
        def _circuit(feature, theta, gate_gens):
            """
            :return: should be full measurement or just 1 measurement,
            so that graddft_qnn.qnn_functional.QNNFunctional.xc_energy works
            """
            # debugging
            # if type(theta) == jax._src.interpreters.ad.JVPTracer:
            #     print(jnp.max(theta).aval, jnp.min(theta).aval, jnp.var(theta).aval, np.mean(theta).aval)   # noqa: E501
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            for idx, gen in enumerate(gate_gens):
                # theta[idx] is ArrayImpl[float]. theta[idx][0] takes the float
                qml.exp(-1j * theta[idx][0] * gen)
            return qml.probs(wires=self.dev.wires)

        self.qnode = qml.QNode(
            _circuit, self.dev, diff_method="backprop", interface="jax-jit"
        )
        self.qnode = jax.jit(self.qnode)

    def circuit(self, feature, theta, gate_gens):
        result = self.qnode(feature, theta, gate_gens)
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

    @staticmethod
    def _twirling(
        ansatz: tuple,
        unitary_reps: list[np.ndarray | qml.operation.Operator],
    ):
        """
        :param ansatz:
        :param unitary_reps: list of all the group member, it should have
        the identity group member
        :return:
        """
        twirled = 0
        for unitary_rep in unitary_reps:
            twirled += unitary_rep @ ansatz @ qml.adjoint(unitary_rep)
            if is_zero_matrix_combination(twirled):
                return None  # Twirling with this group member returns zero matrix!
        twirled /= len(unitary_reps)
        return twirled

    @staticmethod
    def _identity_like(group_member: qml.operation.Operator):
        result = qml.I(0)
        for x in range(1, group_member.num_wires):
            result = qml.prod(result, qml.I(x))
        return result

    @staticmethod
    def _sentence_twirl(sentence: tuple, invariant_rep: list[qml.ops.op_math.Prod]):
        sentence = qml.prod(
            *[getattr(qml, word)(idx) for idx, word in enumerate(sentence)]
        )  # e.g, create qml.X(0) @ qml.Y(1) from X,Y
        return DFTQNN._twirling(sentence, invariant_rep)

    @staticmethod
    def gate_design(
        num_wires: int, invariant_rep: list[np.ndarray | qml.ops.op_math.Prod]
    ) -> tuple[list[str], list[str]]:
        """
        :param num_wires:
        :param invariant_rep: The representation of group members, doesn't have
        identity yet
        :return:
        """
        ansatz_gen = []
        invariant_rep.append(DFTQNN._identity_like(invariant_rep[0]))
        with tqdm(
            total=2**num_wires, desc="Creating invariant gates generator"
        ) as pbar:
            for _, combination in enumerate(
                itertools.product(custom_gates.words.keys(), repeat=num_wires)
            ):
                invariant_gate = DFTQNN._sentence_twirl(combination, invariant_rep)
                if invariant_gate is not None:
                    ansatz_gen.append(invariant_gate)
                    pbar.update()
                    if len(ansatz_gen) == 2**num_wires:
                        break
        return ansatz_gen
