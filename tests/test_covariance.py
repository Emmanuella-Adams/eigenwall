"""
Unit tests for covariance matrix estimation and numerical properties.
"""

import numpy as np
from eigenwall.covariance import sample_covariance, regularize_covariance, check_covariance_properties
from eigenwall.matrix_utils import is_hermitian, is_positive_definite, condition_number


def test_covariance_properties():
    np.random.seed(42)
    N = 4
    T = 1000
    X = np.random.randn(N, T) + 1j * np.random.randn(N, T)

    R = sample_covariance(X)

    # 1. Dimension
    assert R.shape == (N, N)

    # 2. Hermiticity: R == R^H
    assert is_hermitian(R, tol=1e-12)

    # 3. Positive definiteness (for T > N)
    assert is_positive_definite(R, tol=1e-8)

    # 4. Check props dict
    props = check_covariance_properties(R)
    assert props["is_hermitian"] is True
    assert props["is_positive_definite"] is True
    assert props["condition_number"] >= 1.0


def test_regularization():
    R_singular = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
    assert not is_positive_definite(R_singular, tol=1e-8)

    R_reg = regularize_covariance(R_singular, epsilon=0.1)
    assert is_positive_definite(R_reg, tol=1e-8)
    assert condition_number(R_reg) < np.inf
