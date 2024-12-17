import yaml
import pennylane as qml


class DFTQNN:
    def __init__(self, yaml_file):
        self.circuit = None
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
            if "QBIT_PER_AXIS" not in data:
                raise KeyError("YAML file must contain 'QBIT_PER_AXIS' key")
            self.num_qubits = data["QBIT_PER_AXIS"] ** 3

    def circuit(self, theta, phi):
        """
        :param instance: an instance of the class Functional.
        :param rhoinputs: input to the neural network, in the form of an array.
        :return:
        """
        qml.RY(theta, wires=0)
        qml.U1(phi, wires=0)
        return qml.probs()
