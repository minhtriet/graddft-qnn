import itertools

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from flax.typing import Array
from tqdm import tqdm

from graddft_qnn import custom_gates


class DFTQNN(nn.Module):
    dev: qml.device
    ansatz_gen: list[np.array]
    measurements: list[np.array]
    gate_indices: list[int]

    def circuit(self, feature, theta, gate_gens, measurements):
        @qml.qnode(self.dev)
        def _circuit(feature, theta, gate_gens, measurements):
            """
            :param instance: an instance of the class Functional.
            :param rhoinputs: input to the neural network, in the form of an array.
            :return: should be 1 measurement, so that graddft_qnn.qnn_functional.QNNFunctional.xc_energy works

            custom_gates.U2_6_wires(theta, 0)
            return custom_gates.U2_6_wires_measurement(0)
            """
            # if type(theta) == jax._src.interpreters.ad.JVPTracer:
            #     print(jnp.max(theta).aval, jnp.min(theta).aval, jnp.var(theta).aval, np.mean(theta).aval)
            qml.AmplitudeEmbedding(feature, wires=self.dev.wires, pad_with=0.0)
            [
                custom_gates.generate_R_pauli(
                    theta[idx][0], gen
                )  # theta[idx] is ArrayImpl[float]. theta[idx][0] takes the float
                for idx, gen in enumerate(gate_gens)
            ]
            # return [qml.expval(measurements[0])]
            return [qml.expval(measurement) for measurement in measurements]

        result = jnp.array(_circuit(feature, theta, gate_gens, measurements))
        return result

    @nn.compact
    def __call__(self, feature: Array) -> Array:
        theta = self.param(
            "theta",
            nn.initializers.he_normal(),
            # (2 ** len(self.dev.wires), 1),
            (10, 1),
            jnp.float32,
        )
        selected_gates_gen = list(map(lambda i: self.ansatz_gen[i], self.gate_indices))
        return self.circuit(
            feature, theta, selected_gates_gen, list(self.measurements)
        )  # self.ansatz_gen becomes a tuple

    @staticmethod
    def twirling_(ansatz: np.array, unitary_reps: list[np.array]):
        ansatz = ansatz.astype(np.complex64)
        coeffs = []
        for unitary_rep in unitary_reps:
            twirled = 0.5 * (ansatz + unitary_rep @ ansatz @ unitary_rep.conjugate())
            if np.allclose(twirled, np.zeros_like(twirled)):
                return None
            else:
                coeffs.append(qml.pauli_decompose(twirled).coeffs)
        if np.allclose(coeffs, [[1.0]] * len(unitary_reps)):
            return ansatz
        return None

    @staticmethod
    def twirling_2_(ansatz: np.array, unitary_reps: list[np.array]):
        coeffs = []
        for unitary_rep in unitary_reps:
            twirled = 0.5 * (ansatz + unitary_rep @ ansatz @ qml.adjoint(unitary_rep))
            if np.allclose(qml.matrix(twirled), np.zeros_like(qml.matrix(twirled))):
                return None
            else:
                coeffs.append(twirled)
        for i in range(1, len(coeffs)):
            if not np.allclose(qml.matrix(coeffs[i]), qml.matrix(coeffs[0])):
                return None
        return ansatz

    @staticmethod
    def _sentence_twirl(sentence: list[str], invariant_rep: list[np.array]):
        sentence = qml.prod(
            *[getattr(qml, word)(idx) for idx, word in enumerate(sentence)]
        )
        # return DFTQNN.twirling_(matrix, invariant_rep)
        return DFTQNN.twirling_2_(sentence, invariant_rep)

    @staticmethod
    def gate_design(
        num_wires: int, invariant_rep: list[np.array]
    ) -> tuple[list[str], list[str]]:
        switch_threshold = int(np.ceil(2**num_wires / len(custom_gates.words)))
        ansatz_gen = []
        with tqdm(
            total=2**num_wires, desc="Creating invariant gates generator"
        ) as pbar:
            for combination in itertools.product(
                custom_gates.words.keys(), repeat=num_wires
            ):
                if DFTQNN._sentence_twirl(combination, invariant_rep) is not None:
                    if (
                        combination[0]
                        != list(custom_gates.words)[len(ansatz_gen) // switch_threshold]
                    ):
                        # E.g we want 10 ansatz and 3 options (x,y,z) then the first 3 ansatz should start
                        # with x, then next 3 with y, then next with z
                        continue
                    ansatz_gen.append(combination)
                    pbar.update()
                    if len(ansatz_gen) == 2**num_wires:
                        break
        assert len(ansatz_gen) == 2**num_wires
        return ansatz_gen
