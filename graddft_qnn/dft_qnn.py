import yaml
import pennylane as qml


class DFTQNN:
    def __init__(self, yaml_file):
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
            if "QBITS" not in data:
                raise KeyError("YAML file must contain 'QBITS' key")
            self.num_qubits = data["QBITS"]
            self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.theta = [0.0] * self.num_qubits  # todo a different way of init
        self.phi = [0.0] * self.num_qubits
        self.params = {"params": {"phi": self.phi, "theta": self.theta}}

    def _circuit_blueprint(self, feature):
        """
        :param instance: an instance of the class Functional.
        :param rhoinputs: input to the neural network, in the form of an array.
        :return:
        """
        # todo what if nqb is smnaller than feature
        qml.AmplitudeEmbedding(feature, wires=range(self.num_qubits), pad_with=0.0)

        for i in range(self.num_qubits):
            qml.RY(self.theta[i], wires=i)
            qml.U1(self.phi[i], wires=i)
        return qml.probs()

    def circuit(self):
        return qml.QNode(self._circuit_blueprint, self.dev)
