import numpy as np
import pennylane as qml


# single qubit feature map
def angle_feature_map(x, n_qubits, rotaion_axis="X", reps=1):
    for _ in range(reps):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation=rotaion_axis)


def h_angle_feature_map(x, n_qubits, rotaion_axis="Z", reps=1):
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation=rotaion_axis)


# multi qubit feature map
def xx_feature_map(x, n_qubits, reps=1):
    for _ in range(reps):
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)

        for i in range(0, n_qubits - 1):
            qml.IsingXX((np.pi - x[i]) * (np.pi - x[i + 1]), wires=[0, 1])


def yy_feature_map(x, n_qubits, reps=1):
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(x[i], wires=i)

        for i in range(0, n_qubits - 1):
            qml.IsingYY((np.pi - x[i]) * (np.pi - x[i + 1]), wires=[0, 1])


def zz_feature_map(x, n_qubits, reps=3):
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)

        for i in range(0, n_qubits - 1):
            qml.IsingZZ((np.pi - x[i]) * (np.pi - x[i + 1]), wires=[0, 1])
