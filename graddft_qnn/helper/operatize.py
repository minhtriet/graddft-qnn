import pennylane as qml


class Operatize(qml.operation.Operation):
    num_wires = 3
    num_params = 0
    matrix = [[0,0],[0,0]]

    def __init__(self, matrix):
        super().__init__(wires=range(Operatize.num_wires))
        Operatize.matrix = matrix

    @staticmethod
    def compute_matrix():
        return Operatize.matrix

