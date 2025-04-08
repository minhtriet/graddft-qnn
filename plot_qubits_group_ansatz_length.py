import matplotlib.pyplot as plt

# Given dictionary
file_name_params_map = {
    "ansatz_3__180_deg_x_rot_qubits.txt": {
        "qubits": 3,
        "group": "180° x-axis rotation",
    },
    "ansatz_3__90_deg_x_rot]_[_180_deg_x_rot]_[_270_deg_x_rot_qubits.txt": {
        "qubits": 3,
        "group": "C3",
    },
    "ansatz_3__90_deg_x_rot]_[_180_deg_x_rot]_[_270_deg_x_rot]_[_180_deg_y_rot]_[_180_deg_z_rot]_[y_eq_z_rot]_[y_eq_neg_z_rot_qubits.txt": {
        "qubits": 3,
        "group": "D4",
    },
    "ansatz_3__90_deg_x_rot]_[_180_deg_x_rot]_[_270_deg_x_rot]_[_180_deg_y_rot]_[_180_deg_z_rot]_[y_eq_z_rot]_[y_eq_neg_z_rot]_[inversion]_[xy_reflection]_[yz_reflection]_[xz_reflection]_[y_equal_neg_z_reflection]_[y_equal_z_reflection]_[_90_ro_qubits.txt": {
        "qubits": 3,
        "group": "D4h",
    },
    "full_ansatz_6__180_deg_x_rot_qubits.txt": {
        "qubits": 6,
        "group": "180° x-axis rotation",
    },
    "ansatz_6__90_deg_x_rot]_[_180_deg_x_rot]_[_270_deg_x_rot_qubits.txt": {
        "qubits": 6,
        "group": "C3",
    },
    "ansatz_6__90_deg_x_rot]_[_180_deg_x_rot]_[_270_deg_x_rot]_[_180_deg_y_rot]_[_180_deg_z_rot]_[y_eq_z_rot]_[y_eq_neg_z_rot_qubits.txt": {
        "qubits": 6,
        "group": "D4",
    },
    "ansatz_6__90_deg_x_rot]_[_180_deg_x_rot]_[_270_deg_x_rot]_[_180_deg_y_rot]_[_180_deg_z_rot]_[y_eq_z_rot]_[y_eq_neg_z_rot]_[inversion]_[xy_reflection]_[yz_reflection]_[xz_reflection]_[y_equal_neg_z_reflection]_[y_equal_z_reflection]_[_90_ro_qubits.txt": {
        "qubits": 6,
        "group": "D4h",
    },
    "full_ansatz_9__180_deg_x_rot_qubits.txt": {
        "qubits": 9,
        "group": "180° x-axis rotation",
    },
    "full_ansatz_9__90_deg_x_rot]_[_180_deg_x_rot]_[_270_deg_x_rot_qubits.txt": {
        "qubits": 9,
        "group": "C3",
    },
    "full_ansatz_9__90_deg_x_rot]_[_180_deg_x_rot]_[_270_deg_x_rot]_[_180_deg_y_rot]_[_180_deg_z_rot]_[y_eq_z_rot]_[y_eq_neg_z_rot_qubits.txt": {
        "qubits": 9,
        "group": "D4",
    },
    "ansatz_9__90_deg_x_rot]_[_180_deg_x_rot]_[_270_deg_x_rot]_[_180_deg_y_rot]_[_180_deg_z_rot]_[y_eq_z_rot]_[y_eq_neg_z_rot]_[inversion]_[xy_reflection]_[yz_reflection]_[xz_reflection]_[y_equal_neg_z_reflection]_[y_equal_z_reflection]_[_90_ro_qubits.txt": {
        "qubits": 9,
        "group": "D4h",
    },
}


# Function to count number of lines in a file
def count_lines_in_file(file_name):
    try:
        with open(file_name) as f:
            return sum(1 for line in f)
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        return None


# Prepare data by grouping them based on qubits value
qubits_data = {
    3: {"groups": [], "lines": []},
    6: {"groups": [], "lines": []},
    9: {"groups": [], "lines": []},
}

for file_name, params in file_name_params_map.items():
    num_lines = count_lines_in_file(file_name)
    if num_lines is not None:
        qubits = params["qubits"]
        group = params["group"]
        qubits_data[qubits]["groups"].append(group)
        qubits_data[qubits]["lines"].append(num_lines)

# Plotting
plt.figure(figsize=(8, 6))
color = {3: "red", 6: "blue", 9: "green"}
for qubits, data in qubits_data.items():
    plt.plot(
        data["groups"],
        data["lines"],
        marker="o",
        label=f"{qubits} Qubits",
        linestyle="-",
        color=color[qubits],
    )
    for d in range(len(data["groups"])):
        plt.text(y=int(data["lines"][d])*1.2, x=d+0.01, s=data["lines"][d])

plt.xlabel("Group")
plt.ylabel("Number of Ansatz by groups")
plt.yscale("log")
plt.legend(title="Number of Qubits")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
