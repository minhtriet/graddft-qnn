import numpy as np
import pennylane as qml

dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def circuit():
    qml.AmplitudeEmbedding(np.random.random((3,)), wires=[0, 1, 2], pad_with=0.0)
    # Apply the RX gate with a rotation of 2pi/3
    # qml.RX(2 * np.pi / 3, wires=0)
    # qml.RZ(1, wires=0)
    # qml.RX(1, wires=1)
    # qml.RZ(1, wires=2)

    # Measure the qubit in the computational basis
    # return qml.probs(0)
    return qml.probs()


drawer = qml.draw(circuit)
print(drawer())
print(circuit())
