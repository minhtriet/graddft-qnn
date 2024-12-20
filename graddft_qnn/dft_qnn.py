import yaml
import pennylane as qml


class DFTQNN:
    def __init__(self, yaml_file):
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
            if "QBIT_PER_AXIS" not in data:
                raise KeyError("YAML file must contain 'QBIT_PER_AXIS' key")
            self.num_qubits = data["QBIT_PER_AXIS"] ** 3
            self.dev = qml.device("default.qubit", wires=self.num_qubits)

    def _circuit_blueprint(self, theta=0.0, phi=0.0):
        """
        :param instance: an instance of the class Functional.
        :param rhoinputs: input to the neural network, in the form of an array.
        :return:
        """
        for i in range(self.num_qubits):
            qml.RY(theta, wires=i)
            qml.U1(phi, wires=i)
        return qml.probs()

    def circuit(self):
        return qml.QNode(self._circuit_blueprint, self.dev)
