"""
Matrix and statevector numerical utility functions.
"""

from typing import Union
import numpy as np


def normalize_state(vector: np.ndarray) -> np.ndarray:
    """Normalize a complex vector to unit Euclidean norm ||x||_2 = 1."""
    norm = np.linalg.norm(vector)
    if norm < 1e-15:
        raise ValueError("Cannot normalize vector with zero norm")
    return vector / norm


def condition_number(matrix: np.ndarray) -> float:
    """
    Calculate the condition number kappa = lambda_max / lambda_min
    for a positive semi-definite matrix.
    """
    eigvals = np.linalg.eigvalsh(matrix)
    min_eig = eigvals[0]
    max_eig = eigvals[-1]
    if min_eig <= 0:
        return float(np.inf)
    return float(max_eig / min_eig)


def is_hermitian(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if matrix is Hermitian within tolerance: ||A - A^H|| < tol."""
    err = np.linalg.norm(matrix - matrix.conj().T)
    return bool(err < tol)


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if matrix is Hermitian and positive definite (all eigenvalues > tol)."""
    if not is_hermitian(matrix, tol):
        return False
    eigvals = np.linalg.eigvalsh(matrix)
    return bool(np.all(eigvals > tol))
