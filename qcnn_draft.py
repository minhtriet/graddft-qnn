# --------------------------------------------------------------
# Quantum Convolutional Neural Network (QCNN) in Pennylane
# --------------------------------------------------------------

import pennylane as qml
from pennylane import numpy as np

# --------------------- Hyperparameters -----------------------
n_qubits = 8                # Number of qubits (must be power of 2 for pooling)
conv_depth = 2              # Number of conv+pool layers
n_classes = 2               # Two phases: 0 = paramagnetic, 1 = ferromagnetic
shots = 1024                # Number of shots for measurement
seed = 42
np.random.seed(seed)

# --------------------- Quantum Device -----------------------
dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

# --------------------- Encoding -----------------------
def amplitude_encoding(data):
    """Normalize and amplitude-encode classical vector into quantum state."""
    norm = np.sqrt(np.sum(data**2))
    if norm == 0:
        norm = 1.0
    qml.AmplitudeEmbedding(features=data / norm, wires=range(n_qubits), normalize=True)

# --------------------- Convolutional Layer -----------------------
def conv_layer(weights, wires):
    """
    Apply a translationally invariant 2-qubit unitary on neighboring qubits.
    Uses a general 2-qubit ansatz with 15 parameters (full SU(4)).
    """
    for i in range(len(wires) - 1):
        qml.U3(weights[i, 0], weights[i, 1], weights[i, 2], wires=wires[i])
        qml.U3(weights[i, 3], weights[i, 4], weights[i, 5], wires=wires[i+1])
        qml.CNOT(wires=[wires[i], wires[i+1]])
        qml.RZ(weights[i, 6], wires=wires[i])
        qml.RY(weights[i, 7], wires=wires[i+1])
        qml.CNOT(wires=[wires[i+1], wires[i]])
        qml.RY(weights[i, 8], wires=wires[i])
        qml.RZ(weights[i, 9], wires=wires[i+1])
        # Entangling block
        qml.PauliX(wires=wires[i])
        qml.PauliX(wires=wires[i+1])
        qml.CZ(wires=[wires[i], wires[i+1]])
        qml.PauliX(wires=wires[i])
        qml.PauliX(wires=wires[i+1])
        # Final local rotations
        qml.U3(weights[i, 10], weights[i, 11], weights[i, 12], wires=wires[i])
        qml.U3(weights[i, 13], weights[i, 14], weights[i, 14], wires=wires[i+1])

# --------------------- Pooling Layer -----------------------
def pooling_layer(v_weights, wires_in, wires_out):
    """
    Measurement-based pooling: measure one qubit and apply conditional rotation
    on the next to reduce dimension.
    """
    for w_in, w_out in zip(wires_in, wires_out):
        qml.CRY(v_weights[w_out], wires=[w_in, w_out])
        qml.measure(w_in, reset=True)  # Measure and discard

# --------------------- Fully Connected Layer -----------------------
def fc_layer(weights, wires):
    """Variational fully-connected layer on remaining qubits."""
    qml.StronglyEntanglingLayers(weights, wires=wires)

# --------------------- QCNN Circuit -----------------------
@qml.qnode(dev, interface="autograd")
def qcnn_circuit(data, conv_weights, pool_weights, fc_weights):
    # --- Encoding ---
    amplitude_encoding(data)

    active_wires = list(range(n_qubits))

    # --- Convolutional + Pooling Layers ---
    for d in range(conv_depth):
        n_active = len(active_wires)
        if n_active < 2:
            break

        # Convolution on current active wires
        conv_w = conv_weights[d]  # Shape: (n_active-1, 15)
        conv_layer(conv_w, active_wires)

        # Pooling: reduce from n_active -> n_active // 2
        if n_active % 2 != 0:
            # Drop last wire if odd
            active_wires = active_wires[:-1]
            n_active -= 1

        n_pooled = n_active // 2
        wires_in = active_wires[::2]      # Even indices: to be measured
        wires_out = active_wires[1::2]     # Odd indices: survive

        pool_w = pool_weights[d]  # Shape: (n_pooled,)
        pooling_layer(pool_w, wires_in, wires_out)

        active_wires = wires_out  # Update active wires

    # --- Final Fully Connected Layer ---
    if len(active_wires) > 0:
        fc_w = fc_weights  # Shape depends on final wires
        fc_layer(fc_w, active_wires)

    # --- Measurement ---
    return [qml.expval(qml.PauliZ(w)) for w in active_wires[:n_classes]]

# --------------------- Cost & Accuracy -----------------------
def cost(params, X, y):
    conv_weights, pool_weights, fc_weights = params
    predictions = [qcnn_circuit(x, conv_weights, pool_weights, fc_weights) for x in X]
    # Take first expectation as logit for binary classification
    logits = np.array([pred[0] if len(pred) > 0 else 0.0 for pred in predictions])
    labels = 2 * np.array(y) - 1  # Convert {0,1} -> {-1,+1}
    loss = np.mean((logits - labels) ** 2)
    return loss

def accuracy(params, X, y):
    conv_weights, pool_weights, fc_weights = params
    predictions = [qcnn_circuit(x, conv_weights, pool_weights, fc_weights) for x in X]
    logits = np.array([pred[0] if len(pred) > 0 else 0.0 for pred in predictions])
    pred_labels = (logits > 0).astype(int)
    return np.mean(pred_labels == y)

# --------------------- Parameter Initialization -----------------------
def init_params():
    conv_weights = []
    pool_weights = []
    current_wires = n_qubits

    for d in range(conv_depth):
        n_conv = current_wires - 1
        conv_weights.append(np.random.uniform(0, 2*np.pi, size=(n_conv, 15)))
        current_wires = current_wires // 2
        if current_wires % 2 != 0:
            current_wires -= 1
        n_pool = current_wires
        pool_weights.append(np.random.uniform(0, 2*np.pi, size=(n_pool,)))

    # Final FC layer: use StronglyEntanglingLayers template
    final_wires = max(1, n_qubits // (2**conv_depth))
    fc_shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=final_wires)
    fc_weights = np.random.uniform(0, 2*np.pi, size=fc_shape)

    return conv_weights, pool_weights, fc_weights

# --------------------- Generate Dataset (Ising Phases) -----------------------
def ising_ground_state(h):
    """Compute ground state of 1D TFIM: H = -∑ Z_i Z_{i+1} - h ∑ X_i"""
    import scipy.sparse.linalg as sla
    from scipy.sparse import diags

    N = n_qubits
    diagonals = [-h * np.ones(2**N)]
    for i in range(N-1):
        diag = np.zeros(2**N)
        for basis in range(2**N):
            s = [(basis >> j) & 1 for j in range(N)]
            if s[i] == s[i+1]:
                diag[basis] -= 1
            else:
                diag[basis] += 1
        diagonals.append(diag)
    H = sum(diags(diag, 0) for diag in diagonals)
    eigvals, eigvecs = sla.eigsh(H, k=1, which='SA')
    return np.abs(eigvecs.flatten())**2  # Probability vector

# Generate dataset
n_samples = 100
X = []
y = []
for _ in range(n_samples):
    h = np.random.uniform(0.0, 2.0)
    state = ising_ground_state(h)
    X.append(state)
    y.append(0 if h > 1.0 else 1)  # h>1: paramagnetic, h<1: ferromagnetic

X = np.array(X)
y = np.array(y)

# Train/val split
split = int(0.8 * n_samples)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# --------------------- Training -----------------------
params = init_params()
opt = qml.AdamOptimizer(stepsize=0.1)

best_val_acc = 0
best_params = params
patience = 10
wait = 0

print("Starting training...")
for it in range(100):
    batch_idx = np.random.randint(0, len(X_train), size=16)
    X_batch = X_train[batch_idx]
    y_batch = y_train[batch_idx]

    def cost_fn():
        return cost(params, X_batch, y_batch)

    params = opt.step(cost_fn, params)

    if it % 10 == 0:
        train_acc = accuracy(params, X_train, y_train)
        val_acc = accuracy(params, X_val, y_val)
        print(f"Iter {it}: Train Acc = {train_acc:.3f}, Val Acc = {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping!")
                break

# --------------------- Final Evaluation -----------------------
final_train_acc = accuracy(best_params, X_train, y_train)
final_val_acc = accuracy(best_params, X_val, y_val)
print(f"\nFinal Results:")
print(f"Train Accuracy: {final_train_acc:.3f}")
print(f"Validation Accuracy: {final_val_acc:.3f}")

# --------------------- Optional: Draw Circuit -----------------------
print(qml.draw(qcnn_circuit)(X[0], *init_params()))