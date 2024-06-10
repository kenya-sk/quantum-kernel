import dataclasses
from typing import Callable, Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class KernelParams:
    n_qubits: int
    feature_map: Callable
    reps: int
    kernel: Callable
    rotaion_axis: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Dataset:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_cols: list
    target_col: str
