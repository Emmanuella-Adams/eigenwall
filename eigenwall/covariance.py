"""
Covariance matrix estimation, regularization, and spectral analysis for array processing.
"""

from typing import Dict, Any
import numpy as np


def sample_covariance(X: np.ndarray) -> np.ndarray:
    """
    Calculate the sample covariance matrix R = (1 / T) * X @ X^H.

    Args:
        X: Complex snapshot matrix of shape (N, T).

    Returns:
        R: Complex sample covariance matrix of shape (N, N).
    """
    N, T = X.shape
    if T < 1:
        raise ValueError("Number of snapshots T must be >= 1")
    R = (X @ X.conj().T) / float(T)
    # Enforce exact Hermiticity to suppress numerical precision drift
    return 0.5 * (R + R.conj().T)


def regularize_covariance(R: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """
    Diagonal loading / regularization: R_reg = R + epsilon * I.

    Args:
        R: Covariance matrix (N, N).
        epsilon: Regularization parameter (loaded diagonal noise).

    Returns:
        R_reg: Regularized covariance matrix (N, N).
    """
    N = R.shape[0]
    return R + epsilon * np.eye(N, dtype=complex)


def check_covariance_properties(R: np.ndarray, tol: float = 1e-8) -> Dict[str, Any]:
    """
    Verify mathematical properties of the covariance matrix.

    Returns dict with:
        is_hermitian: bool
        hermiticity_error: float
        eigenvalues: np.ndarray (sorted ascending)
        is_positive_definite: bool
        condition_number: float (lambda_max / lambda_min)
    """
    herm_err = float(np.linalg.norm(R - R.conj().T))
    is_herm = herm_err < tol

    # Use eigh for Hermitian matrix eigenvalues
    eigvals = np.linalg.eigvalsh(R)
    min_eig = float(eigvals[0])
    max_eig = float(eigvals[-1])
    is_pd = min_eig > tol
    cond = float(max_eig / min_eig) if min_eig > 1e-15 else np.inf

    return {
        "is_hermitian": is_herm,
        "hermiticity_error": herm_err,
        "eigenvalues": eigvals,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "is_positive_definite": is_pd,
        "condition_number": cond,
    }
