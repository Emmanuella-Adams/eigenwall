"""
Beamforming module: Classical MVDR / Capon beamforming, array pattern computation,
and quantum-to-beamforming weight reconstruction.
"""

from typing import Tuple, Optional, Union
import numpy as np
from eigenwall.signal_model import steering_vector


def mvdr_weights(R: np.ndarray, steering: np.ndarray) -> np.ndarray:
    """
    Compute classical MVDR (Minimum Variance Distortionless Response) weights.

    Solves R * v = a, then normalizes:
    w_MVDR = (R^-1 * a) / (a^H * R^-1 * a)
    ensuring w^H * a = 1 (distortionless constraint toward target).

    Args:
        R: Covariance matrix (N, N).
        steering: Target steering vector a (N,).

    Returns:
        w: Complex weight vector of shape (N,).
    """
    v = np.linalg.solve(R, steering)
    denom = np.vdot(steering, v)  # a^H @ v
    w = v / denom
    return w


def beam_pattern(
    weights: np.ndarray,
    angles_deg: np.ndarray,
    wavelength: float = 1.0,
    spacing: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the array beam pattern B(theta) = |w^H * a(theta)|^2.

    Args:
        weights: Beamformer weights w (N,).
        angles_deg: Array of observation angles in degrees (e.g. -90 to +90).
        wavelength: Carrier wavelength lambda.
        spacing: Element spacing d.

    Returns:
        Tuple of (angles_deg, B_power, B_dB), where:
        - B_power: Raw power pattern |w^H a|^2
        - B_dB: Normalized pattern in dB, relative to peak (max is 0 dB).
    """
    n_elements = len(weights)
    b_power = np.zeros(len(angles_deg), dtype=float)

    for i, theta in enumerate(angles_deg):
        a_theta = steering_vector(n_elements, theta, wavelength, spacing)
        response = np.vdot(weights, a_theta)  # w^H @ a(theta)
        b_power[i] = float(np.abs(response) ** 2)

    max_p = np.max(b_power) if np.max(b_power) > 1e-15 else 1.0
    b_norm = b_power / max_p
    b_db = 10.0 * np.log10(b_norm + 1e-15)

    return angles_deg, b_power, b_db


def quantum_weights_to_beamforming(
    w_quantum_normalized: np.ndarray,
    steering: np.ndarray
) -> np.ndarray:
    """
    Convert a normalized quantum state vector |w_q> to physical MVDR weights.

    A quantum linear system solver outputs |w_q> proportional to R^-1 |a>,
    with unit L2 norm ||w_q|| = 1.
    Physical MVDR beamforming requires the distortionless constraint:
        w^H * a = 1.
    Therefore:
        w_phys = |w_q> / (w_q^H * a)

    Args:
        w_quantum_normalized: Normalized state vector (N,).
        steering: Target steering vector a (N,).

    Returns:
        w_phys: Reconstructed complex weights (N,) satisfying w_phys^H * a = 1.
    """
    # Distortionless constraint requires: w_phys^H * steering = 1.
    # Therefore w_phys = w_q / (steering^H * w_q) = w_q / np.vdot(steering, w_q)
    denom = np.vdot(steering, w_quantum_normalized)
    if np.abs(denom) < 1e-12:
        return w_quantum_normalized / (np.linalg.norm(w_quantum_normalized) + 1e-15)
    return w_quantum_normalized / denom
