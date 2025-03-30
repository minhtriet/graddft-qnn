from enum import Enum


class MetricName(str, Enum):
    DATE = "Date"
    GROUP_MEMBER = "Group members"
    N_QUBITS = "Num qubits"
    N_GATES = "Num gates"
    N_MEASUREMENTS = "Full measurements"
    EPOCHS = "Epochs"
    TEST_LOSS = "Test loss"
