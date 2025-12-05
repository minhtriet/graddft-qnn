import itertools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from flax.typing import Array
from tqdm import tqdm

from graddft_qnn import custom_gates
from graddft_qnn.helper.initialization import batched
from graddft_qnn.unitary_rep import is_zero_matrix_combination, O_h


class DFTQNN(nn.Module):
    dev: qml.device
    ansatz_gen: list[np.array]
    gate_indices: list[int]
    rotate_matrix: np.array = None  # add this to test equivar
    rotate_feature: bool = False
    network_type: str = "qnn"
    # rotate_feature: if True and rotate_matrix, then apply rotation to input feature,
    # otherwise apply rotation to output

    def setup(self) -> None:
        def qnn(feature, theta, gate_gens):
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
                # qml.evolve(gen, theta[idx][0])
            return qml.probs(wires=self.dev.wires)

        def qcnn(feature, theta, gate_gens: dict, ):
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            for layer in range(len(gate_gens)):
                # convolutional layer, where each layer has 1 shared param
                for gen in gate_gens[layer]:
                    qml.exp(-1j * theta[layer][0] * gen)
                # pooling layer
                # which goes from 0 -> num_wires - 0 in layer 0
                # then 1 -> num_wires -1 in layer 1, etc.
                for pool_wires in batched(range(layer, len(self.dev.wires) - layer), 3):
                    O_h.pool(control_wire=pool_wires[0], act_wires=pool_wires[1:], phi=phi[layer][0])

            return qml.probs(wires=self.dev.wires)

        if self.network_type.lower() == "qcnn":
            self.qnode = qml.QNode(qcnn, self.dev)
        elif self.network_type.lower() == "qnn":
            self.qnode = qml.QNode(qnn, self.dev)
        else:
            raise ValueError("unknown network type")
        self.qnode = jax.jit(self.qnode)
        self.selected_gates_gen = list(
            map(lambda i: self.ansatz_gen[i], self.gate_indices)
        )

    def circuit(self, feature, theta, gate_gens):
        if self.rotate_matrix is not None and self.rotate_feature:
            feature = self.rotate_matrix @ feature
        result = self.qnode(feature, theta, gate_gens)
        if self.rotate_matrix is not None and (not self.rotate_feature):
            result = self.rotate_matrix @ result
        return result

    @nn.compact
    def __call__(self, feature: Array) -> Array:
        theta = self.param(
            "theta",
            nn.initializers.he_normal(),
            (len(self.gate_indices), 1),
            jnp.float32,
        )
        return self.circuit(
            feature,
            theta,
            self.selected_gates_gen,
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
    def _identity_like(
        group_member: qml.operation.Operator, wires: list[int] | None = None
    ):
        """
        Generates an identity operation applied to a specified set of wires

        Args:
            group_member (qml.operation.Operator): The quantum operator whose number of wires is used to determine the default wiring.
            wires (list[int] | None, optional): A list of wire indices where the identity operations will be applied.
            If no wires are specified, the identity operation is applied to wires 0, 1 ... group_member's wires.

        Returns:
            qml.operation.Operator: Identity operation applied to the specified wires.

        Example:
            If `group_member.num_wires` is 3 and `wires=[1, 2], 3`, this function would return an identity operation on wires 1, 2, 3.
        """
        if wires is None:
            wires = list(range(group_member.num_wires))
        result = qml.I([wires[0]])
        for x in wires[1:]:
            result = qml.prod(result, qml.I(x))
        return result

    @staticmethod
    def _sentence_twirl(
        sentence: tuple,
        invariant_rep: list[qml.ops.op_math.Prod],
        idx: list[int] | None = None,
    ):
        if not idx:
            idx = list(range(len(sentence)))
        sentence = qml.prod(
            *[getattr(qml, word)(i) for word, i in zip(sentence, idx)]
        )  # e.g, create qml.X(0) @ qml.Y(1) from X,Y with separate idx
        return DFTQNN._twirling(sentence, invariant_rep)

    @staticmethod
    def gate_design(
        invariant_rep: list[np.ndarray | qml.ops.op_math.Prod],
        wires: list[int] | None = None,
    ) -> tuple[list[str], list[str]]:
        """
        :param invariant_rep: The representation of group members, doesn't have
        identity yet
        :return:
        """
        ansatz_gen = []
        invariant_rep.append(DFTQNN._identity_like(invariant_rep[0], wires))
        with tqdm(total=1, desc="Creating invariant gates generator") as pbar:
            for _, combination in enumerate(
                itertools.product(custom_gates.words.keys(), repeat=len(wires))
            ):
                invariant_gate = DFTQNN._sentence_twirl(
                    combination, invariant_rep, wires
                )
                if invariant_gate is not None:
                    ansatz_gen.append(invariant_gate)
                    pbar.update()
                    if len(ansatz_gen) == 3:  # only need 1 gate
                        break
        return ansatz_gen
