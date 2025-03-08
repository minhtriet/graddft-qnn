import numpy as np


class O_h:
    @staticmethod
    def _180_deg_x_rot_matrix(size=2):
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
    def _180_deg_y_rot_matrix(size=2):
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
    def _180_deg_z_rot_matrix(size=2):
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
    def _180_deg_rot_3_axis(size=2):
        return [
            O_h._180_deg_x_rot_matrix(size),
            O_h._180_deg_y_rot_matrix(size),
            O_h._180_deg_z_rot_matrix(size),
            # np.eye(8),
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
