from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm._classes import SVC

from src.kernel import compute_kernel
from src.schema import Dataset, KernelParams


def plot_dataset(
    dataset: Dataset,
    colors: Dict[int, str] = {-1: "red", 1: "green"},
    class_labels: Dict[int, str] = {-1: "-1", 1: "1"},
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(10, 5), tight_layout=True)

    plt.subplot(1, 2, 1)
    for class_value in np.unique(dataset.y_train):
        subset = dataset.x_train[np.where(dataset.y_train == class_value)]
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            c=colors[class_value],
            label=class_labels[class_value],
        )
        plt.xlabel(f"{dataset.feature_cols[0]}")
        plt.ylabel(f"{dataset.feature_cols[1]}")
        plt.legend()
        plt.title("Train Dataset")

    plt.subplot(1, 2, 2)
    for class_value in np.unique(dataset.y_test):
        subset = dataset.x_test[np.where(dataset.y_test == class_value)]
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            c=colors[class_value],
            label=class_labels[class_value],
        )
        plt.xlabel(f"{dataset.feature_cols[0]}")
        plt.ylabel(f"{dataset.feature_cols[1]}")
        plt.legend()
        plt.title("Test Dataset")

    if save_path is not None:
        plt.savefig(save_path)


def plot_predicted_result(
    dataset: Dataset,
    y_pred: np.ndarray,
    colors: Dict[int, str] = {-1: "red", 1: "green"},
    save_path: Optional[str] = None,
):

    plt.figure(figsize=(6, 6))

    for class_value in np.unique(dataset.y_test):
        groud_subset = dataset.x_test[np.where(dataset.y_test == class_value)]
        plt.scatter(
            groud_subset[:, 0],
            groud_subset[:, 1],
            edgecolor=colors[class_value],
            facecolor="none",
            s=100,
            label="Groud Truth",
        )

        predicted_subset = dataset.x_test[np.where(y_pred == class_value)]
        plt.scatter(
            predicted_subset[:, 0],
            predicted_subset[:, 1],
            marker="x",
            c=colors[class_value],
            label="Predicted",
        )

    plt.xlabel(f"{dataset.feature_cols[0]}")
    plt.ylabel(f"{dataset.feature_cols[1]}")
    plt.legend()
    plt.title("Groud Truth VS Predicted (Quantum Kernel)")

    if save_path is not None:
        plt.savefig(save_path)


def plot_decisionon_boundaries(
    model: SVC,
    dataset: Dataset,
    kernel_params: KernelParams,
    step_size: float = 0.1,
    save_path: Optional[str] = None,
):
    x_min, x_max = (
        dataset.x_test[:, 0].min() - step_size,
        dataset.x_test[:, 0].max() + step_size,
    )
    y_min, y_max = (
        dataset.x_test[:, 1].min() - step_size,
        dataset.x_test[:, 1].max() + step_size,
    )
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size)
    )

    # calculate kernel matrix for mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    print(f"Calculating Kernel Matrix for {len(mesh_points)} mesh points")
    K_mesh = compute_kernel(mesh_points, dataset.x_train, kernel_params)

    # Predict the mesh points
    Z = model.predict(K_mesh)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    if np.isnan(Z).any() or np.isinf(Z).any():
        Z = np.nan_to_num(Z, nan=0.0, posinf=1.0, neginf=-1.0)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5, levels=[-1, 0, 1])

    neg_subset = dataset.x_test[np.where(dataset.y_test == -1)]
    plt.scatter(neg_subset[:, 0], neg_subset[:, 1], marker="o", color="black")
    pos_subset = dataset.x_test[np.where(dataset.y_test == 1)]
    plt.scatter(pos_subset[:, 0], pos_subset[:, 1], marker="x", color="black")

    plt.xlabel(f"{dataset.feature_cols[0]}")
    plt.ylabel(f"{dataset.feature_cols[1]}")

    if save_path is not None:
        plt.savefig(save_path)


def plot_metrics(
    reps_list: List[int],
    accuracy_list: List[float],
    target_alignment_list: List[float],
    polarity_list: List[float],
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(12, 4))
    plt.plot(reps_list, accuracy_list, marker="o", label="Accuracy")
    plt.plot(reps_list, target_alignment_list, marker="o", label="Target Alignment")
    plt.plot(reps_list, polarity_list, marker="o", label="Polarity")
    plt.xticks(reps_list)
    plt.legend()
    plt.xlabel("Reps")
    plt.ylabel("Metrics")
    plt.ylim(0.0, 1.05)

    if save_path is not None:
        plt.savefig(save_path)
