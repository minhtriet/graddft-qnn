import numpy as np
import pennylane as qml


class O_h:
    @staticmethod
    def _180_deg_x_rot_matrix(size=2, pauli_word=False) -> np.array:
        if pauli_word:
            n_qubits = np.log2(size**3)
            assert n_qubits.is_integer()
            n_qubits = int(n_qubits)
            num_Is = np.log2(size)
            assert num_Is.is_integer()
            num_Is = int(num_Is)
            prods = [qml.I(i) for i in range(num_Is)]
            prods.extend([qml.X(i) for i in range(num_Is, n_qubits)])
            return qml.prod(*prods)
        # in matrix form
        total_elements = size * size * size
        perm_matrix = np.zeros((total_elements, total_elements), dtype=int)

        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = x
                    new_y = size - 1 - y
                    new_z = size - 1 - z
                    new_idx = new_x * size * size + new_y * size + new_z
                    perm_matrix[orig_idx, new_idx] = 1

        return perm_matrix

    @staticmethod
    def _180_deg_y_rot_matrix(size=2, pauli_word=False):
        """
        2 qubits -> qml.prod(X(0), I(1), X(2)))
        4 qubits -> qml.prod(X(0), X(1), I(2), I(3), X(4), X(5)))
        8 qubits -> qml.prod(X(0), X(1), X(2), I(3), I(4), I(5), X(6), X(7), X(8)))
        """
        if pauli_word:
            n_qubits = np.log2(size**3)
            assert n_qubits.is_integer()
            n_qubits = int(n_qubits)
            num_Is = np.log2(size)
            assert num_Is.is_integer()
            num_Is = int(num_Is)
            prods = (
                [qml.X(i) for i in range(num_Is)]
                + [qml.I(i) for i in range(num_Is, num_Is * 2)]
                + [qml.X(i) for i in range(num_Is * 2, num_Is * 3)]
            )
            return qml.prod(*prods)
        else:
            # y stays the same, x and z change sign
            total_elements = size * size * size
            perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
            for x in range(size):
                for y in range(size):
                    for z in range(size):
                        orig_idx = x * size * size + y * size + z
                        new_x = size - 1 - x
                        new_z = size - 1 - z
                        new_y = y
                        new_idx = new_x * size * size + new_y * size + new_z
                        perm_matrix[orig_idx, new_idx] = 1
            return perm_matrix

    @staticmethod
    def _180_deg_z_rot_matrix(size=2, pauli_word=False):
        """
        2 qubits -> qml.prod(X(0), X(1), I(2)))
        4 qubits -> qml.prod(X(0), X(1), X(2), X(3), I(4), I(5)))
        8 qubits -> qml.prod(X(0), X(1), X(2), X(3), X(4), X(5), I(6), I(7), I(8)))
        """
        if pauli_word:
            n_qubits = np.log2(size**3)
            assert n_qubits.is_integer()
            n_qubits = int(n_qubits)
            num_Is = np.log2(size)
            assert num_Is.is_integer()
            num_Is = int(num_Is)
            prods = (
                [qml.X(i) for i in range(num_Is)]
                + [qml.X(i) for i in range(num_Is, num_Is * 2)]
                + [qml.I(i) for i in range(num_Is * 2, num_Is * 3)]
            )
            return qml.prod(*prods)
        else:
            # z stays the same, x and y change sign
            total_elements = size * size * size
            perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
            for x in range(size):
                for y in range(size):
                    for z in range(size):
                        orig_idx = x * size * size + y * size + z
                        new_z = z
                        new_x = size - 1 - x
                        new_y = size - 1 - y
                        new_idx = new_x * size * size + new_y * size + new_z
                        perm_matrix[orig_idx, new_idx] = 1

            return perm_matrix

    @staticmethod
    def _180_deg_rot():
        return [O_h._180_deg_z_rot_matrix(), np.eye(8)]

    @staticmethod
    def C2_group(size=2, pauli_word=False):
        return [
            O_h._180_deg_x_rot_matrix(size, pauli_word),
            O_h._180_deg_y_rot_matrix(size, pauli_word),
            O_h._180_deg_z_rot_matrix(size, pauli_word),
        ]

    @staticmethod
    def _180_deg_rot_ref(size=2):
        return [
            O_h._180_deg_x_rot_matrix(size),
            O_h._180_deg_y_rot_matrix(size),
            O_h._180_deg_z_rot_matrix(size),
            # np.eye(8),
            O_h.reflection_yz(),
        ]

    @staticmethod
    def rz():
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    @staticmethod
    def reflection_yz():
        return np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
