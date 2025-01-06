import numpy as np
import pennylane as qml

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def circuit():
    # Apply the RX gate with a rotation of 2pi/3
    # qml.RX(2 * np.pi / 3, wires=0)
    qml.RX(4 * np.pi / 3, wires=0)
    qml.RX(4 * np.pi / 3, wires=1)

    # Measure the qubit in the computational basis
    # return qml.probs(0)
    return qml.state()


print(circuit())
