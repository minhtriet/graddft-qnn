import matplotlib.pyplot as plt

# Given dictionary
file_name_params_map = {
    "full_ansatz_6__180_deg_x_rot_qubits.txt": {"qubits": 6, "group": "180° x-axis rotation"},
    "full_ansatz_9__180_deg_x_rot_qubits.txt": {"qubits": 9, "group": "180° x-axis rotation"},
    "ansatz_6__90_deg_x_rot]_[_180_deg_x_rot]_[_270_deg_x_rot]_[_180_deg_y_rot]_[_180_deg_z_rot]_[y_eq_z_rot]_[y_eq_neg_z_rot]_[inversion]_[xy_reflection]_[yz_reflection]_[xz_reflection]_[y_equal_neg_z_reflection]_[y_equal_z_reflection]_[_90_ro_qubits.txt": {"qubits": 6, "group": "D4h"},
    "full_ansatz_9__111_deg_x_rot_qubits.txt": {"qubits": 9, "group": "abcd"},
}

# Function to count number of lines in a file
def count_lines_in_file(file_name):
    try:
        with open(file_name, 'r') as f:
            return sum(1 for line in f)
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        return None

# Prepare data by grouping them based on qubits value
qubits_data = {6: {"groups": [], "lines": []}, 9: {"groups": [], "lines": []}}

for file_name, params in file_name_params_map.items():
    num_lines = count_lines_in_file(file_name)
    if num_lines is not None:
        qubits = params["qubits"]
        group = params["group"]
        qubits_data[qubits]["groups"].append(group)
        qubits_data[qubits]["lines"].append(num_lines)

# Plotting
plt.figure(figsize=(8, 6))

# Line plots for each qubits value (6 and 9)
for qubits, data in qubits_data.items():
    plt.plot(data["groups"], data["lines"], marker='o', label=f'{qubits} Qubits', linestyle='-', color='blue' if qubits == 6 else 'green')

# Adding labels and title
plt.xlabel('Group')
plt.ylabel('Number of Lines in File')
plt.title('Number of Lines in File by Group')

# Add a legend
plt.legend(title='Number of Qubits')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
