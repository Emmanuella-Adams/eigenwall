"""
Evaluation metrics for quantum linear solver and adaptive beamforming performance.
"""

from typing import Optional
import numpy as np
from eigenwall.signal_model import steering_vector
from eigenwall.matrix_utils import normalize_state


def state_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """
    Compute quantum state fidelity between two states:
        F = |<a|b>|^2 / (||a||^2 * ||b||^2)

    For pure quantum states, F in [0, 1].
    """
    a_norm = normalize_state(state_a)
    b_norm = normalize_state(state_b)
    overlap = np.vdot(a_norm, b_norm)
    return float(np.abs(overlap) ** 2)


def residual_error(matrix: np.ndarray, vector_b: np.ndarray, solution_x: np.ndarray) -> float:
    """
    Compute relative linear system residual error:
        ||A * x - b||_2 / ||b||_2

    Note: if solution_x is a normalized state |x>, scale it by least-squares:
        alpha = b^H A x / ||A x||^2, then compute ||alpha A x - b|| / ||b||.
    """
    b_norm = np.linalg.norm(vector_b)
    if b_norm < 1e-15:
        return 0.0

    Ax = matrix @ solution_x
    # Optimal scaling factor to align magnitudes
    alpha = np.vdot(Ax, vector_b) / (np.vdot(Ax, Ax) + 1e-15)
    residual = np.linalg.norm(alpha * Ax - vector_b)
    return float(residual / b_norm)


def array_response_power(
    weights: np.ndarray,
    angle_deg: float,
    wavelength: float = 1.0,
    spacing: Optional[float] = None
) -> float:
    """Calculate raw beamformer power gain at a specified angle |w^H a(theta)|^2."""
    n_elements = len(weights)
    a = steering_vector(n_elements, angle_deg, wavelength, spacing)
    resp = np.vdot(weights, a)
    return float(np.abs(resp) ** 2)


def null_depth_db(
    weights: np.ndarray,
    target_angle: float,
    interference_angle: float,
    wavelength: float = 1.0,
    spacing: Optional[float] = None
) -> float:
    """
    Calculate null depth in dB:
        Null Depth = 10 * log10( P(target_angle) / P(interference_angle) )
    Higher value indicates stronger interference suppression relative to target.
    """
    p_target = array_response_power(weights, target_angle, wavelength, spacing)
    p_interf = array_response_power(weights, interference_angle, wavelength, spacing)
    if p_interf < 1e-15:
        return 100.0  # Cap at +100 dB for numerical zero
    return float(10.0 * np.log10(p_target / p_interf))


def beam_direction_error(
    weights_ref: np.ndarray,
    weights_test: np.ndarray,
    angles_deg: Optional[np.ndarray] = None,
    wavelength: float = 1.0,
    spacing: Optional[float] = None
) -> float:
    """
    Calculate angular difference between peak beam directions:
        Delta theta = |theta_peak(test) - theta_peak(ref)|
    """
    if angles_deg is None:
        angles_deg = np.linspace(-90.0, 90.0, 721)

    n_elements = len(weights_ref)
    p_ref = [np.abs(np.vdot(weights_ref, steering_vector(n_elements, th, wavelength, spacing))) ** 2 for th in angles_deg]
    p_test = [np.abs(np.vdot(weights_test, steering_vector(n_elements, th, wavelength, spacing))) ** 2 for th in angles_deg]

    peak_ref = angles_deg[np.argmax(p_ref)]
    peak_test = angles_deg[np.argmax(p_test)]

    return float(np.abs(peak_test - peak_ref))


def compute_sinr_db(
    weights: np.ndarray,
    steering_target: np.ndarray,
    R_interf_noise: np.ndarray,
    target_power: float = 1.0
) -> float:
    """
    Compute Output Signal-to-Interference-plus-Noise Ratio (SINR) in dB:
        SINR = (P_s * |w^H * a_s|^2) / (w^H * R_{i+n} * w)
    """
    sig_out = target_power * float(np.abs(np.vdot(weights, steering_target)) ** 2)
    in_out = float(np.real(np.vdot(weights, R_interf_noise @ weights)))
    if in_out < 1e-15:
        return 100.0
    return float(10.0 * np.log10(sig_out / in_out))


def multi_null_depth_db(
    weights: np.ndarray,
    target_angle: float,
    interference_angles: list,
    wavelength: float = 1.0,
    spacing: Optional[float] = None
) -> dict:
    """Compute null depths in dB across multiple simultaneous interference angles."""
    results = {}
    for angle_i in interference_angles:
        nd = null_depth_db(weights, target_angle, angle_i, wavelength, spacing)
        results[angle_i] = nd
    return results
