from typing import Callable

import numpy as np
import pennylane as qml
from pennylane.measurements.probs import ProbabilityMP
from tqdm import tqdm

from src.schema import KernelParams


def kernel(
    x1: np.ndarray, x2: np.ndarray, kernel_params: KernelParams
) -> ProbabilityMP:
    if kernel_params.rotaion_axis is not None:
        kernel_params.feature_map(
            x1, kernel_params.n_qubits, kernel_params.rotaion_axis, kernel_params.reps
        )
        qml.adjoint(kernel_params.feature_map)(
            x2, kernel_params.n_qubits, kernel_params.rotaion_axis, kernel_params.reps
        )
    else:
        kernel_params.feature_map(x1, kernel_params.n_qubits, kernel_params.reps)
        qml.adjoint(kernel_params.feature_map)(
            x2, kernel_params.n_qubits, kernel_params.reps
        )

    return qml.probs(wires=range(kernel_params.n_qubits))


def compute_kernel(
    x_array_1: np.ndarray, x_array_2: np.ndarray, kernel_params: KernelParams
) -> np.ndarray:
    n_samples_1 = len(x_array_1)
    n_samples_2 = len(x_array_2)
    kernel_matrix = np.zeros((n_samples_1, n_samples_2))

    for i in tqdm(range(n_samples_1)):
        for j in range(n_samples_2):
            kernel_matrix[i, j] = kernel_params.kernel(
                x_array_1[i], x_array_2[j], kernel_params
            )[
                0
            ]  # get |00> state probability

    return kernel_matrix


def evaluate_kernel(
    x_train: np.ndarray, y_train: np.ndarray, kernel_params: KernelParams
) -> tuple[float, float]:
    target_alignment = qml.kernels.target_alignment(
        x_train, y_train, lambda x1, x2: kernel_params.kernel(x1, x2, kernel_params)[0]
    )
    polarity = qml.kernels.polarity(
        x_train, y_train, lambda x1, x2: kernel_params.kernel(x1, x2, kernel_params)[0]
    )
    print(f"Target Kernel Alignment: {target_alignment:.2f}")
    print(f"Kernel Polarity: {polarity:.2f}")

    return target_alignment, polarity
