import numpy as np


class O_h:
    @staticmethod
    def _180_deg_rot():
        return [O_h._180_deg_z_rot_matrix(), np.eye(8)]

    @staticmethod
    def _180_deg_rot_3_axis():
        return [
            O_h._180_deg_x_rot_matrix(),
            O_h._180_deg_y_rot_matrix(),
            O_h._180_deg_z_rot_matrix(),
            # np.eye(8),
        ]

    @staticmethod
    def _180_deg_rot_ref():
        return [
            O_h._180_deg_x_rot_matrix(),
            O_h._180_deg_y_rot_matrix(),
            O_h._180_deg_z_rot_matrix(),
            # np.eye(8),
            O_h.reflection_yz(),
        ]

    @staticmethod
    def rz():
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    @staticmethod
    def _180_deg_z_rot_matrix():
        return np.array(
            [
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
            ]
        )

    @staticmethod
    def _180_deg_y_rot_matrix():
        return np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
            ]
        )

    @staticmethod
    def _180_deg_x_rot_matrix():
        return np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )

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
