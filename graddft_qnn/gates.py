import pennylane as qml

# ====================================================================
#     Legacy
# ====================================================================
@staticmethod
def U_O3(psi, theta, phi, wires, gamma=1.23):
    # todo change gamma to a learnable param
    qml.RZ(psi, wires=wires[0])
    qml.RX(theta, wires=wires[0])
    qml.RZ(phi, wires=wires[0])

    qml.RZ(psi, wires=wires[1])
    qml.RX(theta, wires=wires[1])
    qml.RZ(phi, wires=wires[1])

    qml.RZ(psi, wires=wires[2])
    qml.RX(theta, wires=wires[2])
    qml.RZ(phi, wires=wires[2])

    # todo loss function
    qml.QubitUnitary(DFTQNN._RXXX_matrix(gamma), wires=wires[0:3])


def V_O3(self, psi, theta, phi):
    pass

def U1_AE(self, thetas, wires):
    # circuit 1
    qml.Rx(thetas[0], wires=wires[0])
    qml.Rx(thetas[1], wires=wires[1])
    qml.QubitUnitary(self._RXX_matrix(thetas[2]), wires=wires[0, 1])

def U3_AE(self, theta_1, theta_2, theta_3, theta_4, theta_5, wires):
    # circuit 4
    qml.Rot(theta_1, theta_2, theta_3, wires=wires[0])
    qml.RX(theta_4, wires=wires[1])
    qml.QubitUnitary(self._RXX_matrix(theta_5), wires=wires)

def V1_AE(self, phi_1, phi_2, control_wire, wire):
    # circuit 6
    qml.CRX(phi_1, wires=[control_wire, wire])
    qml.X(control_wire)
    qml.CRX(phi_2, wires=[control_wire, wire])
    qml.X(control_wire)

def V3_AE(self, phi_1, phi_2, control_wire, wire):
    # circuit 7
    qml.CRZ(phi_1, wires=[control_wire, wire])
    qml.X(control_wire)
    qml.CRY(phi_2, wires=[control_wire, wire])
    qml.X(control_wire)

def _X_matrix(self):
    return np.array([[0, 1], [1, 0]])

def _RXX_matrix(self, theta):
    return np.cos(theta * 0.5) * np.eye(4) - 1j * np.sin(theta * 0.5) * np.kron(
        self._X_matrix(), self._X_matrix()
    )
