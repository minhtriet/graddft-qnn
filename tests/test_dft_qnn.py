import functools

import numpy as np

from graddft_qnn.custom_gates import words
from graddft_qnn.dft_qnn import DFTQNN
from graddft_qnn.unitary_rep import O_h

x_rot_matrix = np.array(
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
y_rot_matrix = np.array([])
z_rot_matrix = np.array(
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
x_reflect_matrix = []
y_reflect_matrix = []
z_reflect_matrix = []


def test_twirling():
    sentence = ["X"] * 6
    sentence_matrix = [words[x] for x in sentence]
    matrix = functools.reduce(np.kron, sentence_matrix)
    size = np.cbrt(matrix.shape[0])
    assert size.is_integer()
    assert np.allclose(matrix, DFTQNN.twirling_2_(matrix, O_h.C2_group(int(size))))

    sentence = ["X", "X", "X", "X", "I", "Z"]
    sentence_matrix = [words[x] for x in sentence]
    matrix = functools.reduce(np.kron, sentence_matrix)
    assert DFTQNN.twirling_2_(matrix, O_h.C2_group(int(size))) is None


def test_gate_design():
    gate_gen = DFTQNN.gate_design(6, O_h.C2_group(4, pauli_word=True))
    gate_gen = ["".join(g) for g in gate_gen]
    assert gate_gen == [
        "XXXXXX",
        "XXXXXI",
        "XXXXYY",
        "XXXXYZ",
        "XXXXZY",
        "XXXXZZ",
        "XXXXIX",
        "XXXXII",
        "XXXIXX",
        "XXXIXI",
        "XXXIYY",
        "XXXIYZ",
        "XXXIZY",
        "XXXIZZ",
        "XXXIIX",
        "XXXIII",
        "XXYYXX",
        "XXYYXI",
        "XXYYYY",
        "XXYYYZ",
        "XXYYZY",
        "XXYYZZ",
        "XXYYIX",
        "XXYYII",
        "XXYZXX",
        "XXYZXI",
        "XXYZYY",
        "XXYZYZ",
        "XXYZZY",
        "XXYZZZ",
        "XXYZIX",
        "XXYZII",
        "XXZYXX",
        "XXZYXI",
        "XXZYYY",
        "XXZYYZ",
        "XXZYZY",
        "XXZYZZ",
        "XXZYIX",
        "XXZYII",
        "XXZZXX",
        "XXZZXI",
        "XXZZYY",
        "XXZZYZ",
        "XXZZZY",
        "XXZZZZ",
        "XXZZIX",
        "XXZZII",
        "XXIXXX",
        "XXIXXI",
        "XXIXYY",
        "XXIXYZ",
        "XXIXZY",
        "XXIXZZ",
        "XXIXIX",
        "XXIXII",
        "XXIIXX",
        "XXIIXI",
        "XXIIYY",
        "XXIIYZ",
        "XXIIZY",
        "XXIIZZ",
        "XXIIIX",
        "XXIIII",
    ]
