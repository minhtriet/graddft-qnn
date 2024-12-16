import yaml

class DFTQNN:
    def __init__(self, yaml_file):
        self.circuit = None
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
            if 'num_qubits' not in data:
                raise KeyError("YAML file must contain 'num_qubits' key")
            self.num_qubits = data['num_qubits']
        self.dev = qml.device('default.qubit', wires=self.num_qubits)


    @qml.qnode(self.dev)
    def coefficients(instance, rhoinputs):
        """
        :param instance: an instance of the class Functional.
        :param rhoinputs: input to the neural network, in the form of an array.
        :return:
        """

        qml.RX(x, wires=0)
        qml.U1(theta)
        return qml.state()

