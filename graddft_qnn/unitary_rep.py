import logging
from itertools import product

import numpy as np
import pennylane as qml
from scipy.sparse import csr_matrix


class O_h:
    @staticmethod
    def _180_deg_x_rot_sparse(
        size=2, pauli_word=True, starting_wire=0
    ) -> csr_matrix | qml.SparseHamiltonian:
        """
        Allowing starting_wire to be set so that this can be used when number of qubits is not the number of wires
        this gate acts on.
        """
        # 3 qubits I(0) @ X(1) @ X(2)
        # 6 qubits I(0) @ I(1) @ X(2) @ X(3) @ X(4) @ X(5)
        # 9 qubits I(0) @ I(1) @ I(2) @ X(3) @ X(4) @ X(5) @ X(6) @ X(7) @ X(8)
        if pauli_word:
            n_qubits = np.log2(size**3)
            assert n_qubits.is_integer()
            n_qubits = int(n_qubits)
            num_Is = np.log2(size)
            assert num_Is.is_integer()
            num_Is = int(num_Is)
            prods = [qml.I(i) for i in range(starting_wire, num_Is + starting_wire)] + [
                qml.X(i)
                for i in range(num_Is + starting_wire, n_qubits + starting_wire)
            ]
            return qml.prod(*prods)
        total_elements = size * size * size
        row_indices = []
        col_indices = []
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = x
                    new_y = size - 1 - y
                    new_z = size - 1 - z
                    new_idx = new_x * size * size + new_y * size + new_z
                    row_indices.append(orig_idx)
                    col_indices.append(new_idx)
        perm_matrix = csr_matrix(
            ([1] * len(row_indices), (row_indices, col_indices)),
            shape=(total_elements, total_elements),
            dtype=int,
        )
        return qml.SparseHamiltonian(
            perm_matrix, wires=range(int(np.log2(total_elements)))
        )

    @staticmethod
    def _180_deg_x_rot(size=2, pauli_word=False, starting_wire=0) -> np.array:
        if pauli_word:
            n_qubits = np.log2(size**3)
            assert n_qubits.is_integer()
            n_qubits = int(n_qubits)
            num_Is = np.log2(size)
            assert num_Is.is_integer()
            num_Is = int(num_Is)
            prods = [qml.I(i) for i in range(starting_wire, starting_wire + num_Is)]
            prods.extend(
                [
                    qml.X(i)
                    for i in range(starting_wire + num_Is, starting_wire + n_qubits)
                ]
            )
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
    def _180_deg_y_rot(size=2, pauli_word=True, starting_wire=0):
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
                [qml.X(i) for i in range(starting_wire, starting_wire + num_Is)]
                + [
                    qml.I(i)
                    for i in range(starting_wire + num_Is, starting_wire + num_Is * 2)
                ]
                + [
                    qml.X(i)
                    for i in range(
                        starting_wire + num_Is * 2, starting_wire + num_Is * 3
                    )
                ]
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
    def _180_deg_z_rot(size=2, pauli_word=True, starting_wire=0):
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
            prods = [
                qml.X(i) for i in range(starting_wire, starting_wire + num_Is * 2)
            ] + [
                qml.I(i)
                for i in range(starting_wire + num_Is * 2, starting_wire + num_Is * 3)
            ]
            return qml.prod(*prods)
        else:
            logging.warning("Deprecated: use pauli_word=True version for large size")
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
    def C2_group(
        size=2, pauli_word=False
    ) -> list[qml.Hamiltonian | qml.pauli.PauliSentence]:
        return [
            O_h._180_deg_x_rot(size, pauli_word),
            O_h._180_deg_y_rot(size, pauli_word),
            O_h._180_deg_z_rot(size, pauli_word),
        ]

    @staticmethod
    def _270_deg_x_rot(
        size=2, pauli_word=True
    ) -> np.ndarray | qml.Hamiltonian | qml.pauli.PauliSentence:
        total_elements = size * size * size
        perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = x
                    new_z = y
                    new_y = size - 1 - z
                    new_idx = new_x * size * size + new_y * size + new_z
                    perm_matrix[orig_idx, new_idx] = 1
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def _270_deg_x_rot_sparse(size=2, pauli_word=True):
        total_elements = size * size * size
        row_indices = []
        col_indices = []
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = x
                    new_z = y
                    new_y = size - 1 - z
                    new_idx = new_x * size * size + new_y * size + new_z
                    row_indices.append(orig_idx)
                    col_indices.append(new_idx)
        perm_matrix = csr_matrix(
            ([1] * len(row_indices), (row_indices, col_indices)),
            shape=(total_elements, total_elements),
            dtype=int,
        )
        if pauli_word:
            return qml.SparseHamiltonian(
                perm_matrix, wires=range(int(np.log2(total_elements)))
            )
        else:
            return perm_matrix

    @staticmethod
    def _90_deg_x_rot(size=2, pauli_word=False):
        total_elements = size * size * size
        perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = x
                    new_y = z
                    new_z = size - 1 - y
                    new_idx = new_x * size * size + new_y * size + new_z
                    perm_matrix[orig_idx, new_idx] = 1
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def _90_deg_x_rot_sparse(size=2, pauli_word=True):
        total_elements = size * size * size
        row_indices, col_indices = [], []
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = x
                    new_y = z
                    new_z = size - 1 - y
                    new_idx = new_x * size * size + new_y * size + new_z
                    row_indices.append(orig_idx)
                    col_indices.append(new_idx)
        perm_matrix = csr_matrix(
            ([1] * len(row_indices), (row_indices, col_indices)),
            shape=(total_elements, total_elements),
            dtype=int,
        )
        if pauli_word:
            return qml.SparseHamiltonian(
                perm_matrix, wires=range(int(np.log2(total_elements)))
            )
        else:
            return perm_matrix

    @staticmethod
    def _180_deg_rot_ref(size=2):
        return [
            O_h._180_deg_x_rot(size),
            O_h._180_deg_y_rot(size),
            O_h._180_deg_z_rot(size),
            # np.eye(8),
            O_h.yz_reflection(),
        ]

    @staticmethod
    def xy_reflection(size=2, pauli_word=False):
        total_elements = size * size * size
        perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = x
                    new_y = y
                    new_z = size - z - 1
                    new_idx = new_x * size * size + new_y * size + new_z
                    perm_matrix[orig_idx, new_idx] = 1
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def yz_reflection(size=2, pauli_word=False):
        total_elements = size * size * size
        perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_y = y
                    new_z = z
                    new_x = size - x - 1
                    new_idx = new_x * size * size + new_y * size + new_z
                    perm_matrix[orig_idx, new_idx] = 1
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def xz_reflection(size=2, pauli_word=False):
        total_elements = size * size * size
        perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = x
                    new_z = z
                    new_y = size - y - 1
                    new_idx = new_x * size * size + new_y * size + new_z
                    perm_matrix[orig_idx, new_idx] = 1
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def y_equal_neg_z_reflection(size=2, pauli_word=False):
        total_elements = size * size * size
        perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = x
                    new_y = z
                    new_z = y
                    new_idx = new_x * size * size + new_y * size + new_z
                    perm_matrix[orig_idx, new_idx] = 1
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def y_equal_z_reflection(size=2, pauli_word=False):
        total_elements = size * size * size
        perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = x
                    new_y = size - z - 1
                    new_z = size - y - 1
                    new_idx = new_x * size * size + new_y * size + new_z
                    perm_matrix[orig_idx, new_idx] = 1
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def _90_roto_x_reflect_yz(size=2, pauli_word=False):
        perm_matrix = O_h._90_deg_x_rot(size) @ O_h.yz_reflection(size)
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def _270_roto_x_reflect_yz(size=2, pauli_word=False):
        perm_matrix = O_h._270_deg_x_rot(size) @ O_h.yz_reflection(size)
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def y_eq_z_rot(size=2, pauli_word=False):
        total_elements = size * size * size
        perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = size - 1 - x
                    new_y = size - 1 - z
                    new_z = size - 1 - y
                    new_idx = new_x * size * size + new_y * size + new_z
                    perm_matrix[orig_idx, new_idx] = 1
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def y_eq_neg_z_rot(size=2, pauli_word=False):
        total_elements = size * size * size
        perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = size - 1 - x
                    new_z = y
                    new_y = z
                    new_idx = new_x * size * size + new_y * size + new_z
                    perm_matrix[orig_idx, new_idx] = 1
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def inversion(size=2, pauli_word=False):
        total_elements = size * size * size
        perm_matrix = np.zeros((total_elements, total_elements), dtype=int)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    orig_idx = x * size * size + y * size + z
                    new_x = size - 1 - x
                    new_y = size - 1 - y
                    new_z = size - 1 - z
                    new_idx = new_x * size * size + new_y * size + new_z
                    perm_matrix[orig_idx, new_idx] = 1
        if pauli_word:
            return qml.pauli_decompose(
                perm_matrix, check_hermitian=False, hide_identity=True
            )
        else:
            return perm_matrix

    @staticmethod
    def pool(control_wire, act_wires, phi):
        assert phi.shape == (2,), "Angle parameter phi must be of shape (2,)"
        base1 = qml.prod(*[qml.RX(phi[0], wires=x) for x in act_wires])
        controlled1 = qml.ops.op_math.Controlled(
            base1, control_wires=control_wire, control_values=True
        )
        base2 = qml.prod(*[qml.RX(phi[1], wires=x) for x in act_wires])
        controlled2 = qml.ops.op_math.Controlled(
            base2, control_wires=control_wire, control_values=False
        )
        return controlled1, controlled2

def is_group(matrices: list[np.ndarray], group_name: list[str]) -> bool:
    """
    Check if a list of matrices forms a group.

    Args:
        matrices: List of numpy arrays representing matrices

    Returns:
        bool: True if matrices form a group, False otherwise
    """
    group_products = list(product(group_name, group_name))
    size = matrices[0].shape[0]
    for i, mat_tuple in enumerate(product(matrices, matrices)):
        product_matrix = mat_tuple[0] @ mat_tuple[1]
        if not any(np.allclose(product_matrix, M) for M in matrices + [np.eye(size)]):
            logging.info(
                f"Closure failed: Product {group_products[i]} not in the group"
            )
            return False

    for i, matrix in enumerate(matrices):
        inverse = np.linalg.inv(matrix)
        if not any(np.allclose(inverse, M) for M in matrices):
            logging.info(f"Inverse not found for matrix: {group_name[i]}")
            return False

    return True


def is_zero_matrix_combination(op: qml.operation.Operator):
    try:
        data = op.sparse_matrix().data
    except qml.operation.SparseMatrixUndefinedError:
        data = qml.matrix(op)
    return np.allclose(np.zeros_like(data), data)
