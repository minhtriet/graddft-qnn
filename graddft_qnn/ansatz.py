import functools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pennylane as qml

from graddft_qnn.custom_gates import ZZZ_matrix


@dataclass
class Ansatz(qml.operation.Operation):
    """
    XYZ
    XYZ --- ZZZ
    XYZ
    equation 55 JJMeyer
    """

    num_wires = 3
    num_params = 0
    _wire_to_single_qubit_gates = {
        (0,): [qml.X.compute_matrix(), qml.Y.compute_matrix(), qml.Z.compute_matrix()],
        (1,): [qml.X.compute_matrix(), qml.Y.compute_matrix(), qml.Z.compute_matrix()],
        (2,): [qml.X.compute_matrix(), qml.Y.compute_matrix(), qml.Z.compute_matrix()],
    }
    _wire_to_triple_qubit_gates = {(0, 1, 2): [ZZZ_matrix()]}

    @property
    def wire_to_single_qubit_gates(self) -> float:
        """
        Return representation in a multiple qubits environment
        :return:
        """
        result = defaultdict(lambda: [])
        for wires, gates in self._wire_to_single_qubit_gates.items():
            # reduce list of gates to a matrix repr
            size = 1
            for gate in gates:
                multi_qubit_repr = []
                for i in range(Ansatz.num_wires):
                    if i == wires[0]:
                        multi_qubit_repr.append(gate)
                        size *= gate.shape[0]
                    else:
                        multi_qubit_repr.append(np.eye(2))
                        size *= 2
                    if size == 2**Ansatz.num_wires:
                        break
                result[wires].append(functools.reduce(np.kron, multi_qubit_repr))
        return result

    @property
    def wire_to_triple_qubit_gates(self):
        return self._wire_to_triple_qubit_gates
