from math import sqrt
import pennylane as qml

dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def circuit(f=None):
    qml.AmplitudeEmbedding(features=f, wires=range(4), pad_with=0.)
    qml.U1(0.1, wires=0)
    return qml.expval(qml.Z(0)), qml.state(), qml.probs()

res, state, probs = circuit(f=[1/sqrt(2)]*16)
print(res)
print(state)
print(probs)
