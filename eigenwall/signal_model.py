"""
Signal model module for uniform linear antenna arrays in deep-space communications.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
import numpy as np


@dataclass
class ArrayConfig:
    """Configuration for a Uniform Linear Array (ULA)."""
    n_elements: int = 4
    wavelength: float = 1.0
    spacing: Optional[float] = None

    def __post_init__(self):
        if self.spacing is None:
            self.spacing = self.wavelength / 2.0


def steering_vector(
    n_elements: int,
    angle_deg: float,
    wavelength: float = 1.0,
    spacing: Optional[float] = None
) -> np.ndarray:
    """
    Calculate the steering vector for a Uniform Linear Array (ULA).

    a_n(theta) = exp(j * n * k * d * sin(theta))
    where k = 2 * pi / wavelength, d = element spacing, n = 0, ..., N-1.

    Args:
        n_elements: Number of antenna elements N.
        angle_deg: Angle of arrival in degrees (broadside is 0 deg).
        wavelength: Carrier wavelength lambda (meters).
        spacing: Inter-element spacing d (default: lambda / 2).

    Returns:
        Steering vector a(theta) as complex ndarray of shape (n_elements,).
    """
    if spacing is None:
        spacing = wavelength / 2.0

    k = 2.0 * np.pi / wavelength
    theta_rad = np.deg2rad(angle_deg)
    n = np.arange(n_elements)
    phase = n * k * spacing * np.sin(theta_rad)
    return np.exp(1j * phase)


def generate_sinusoid(
    frequency: float,
    n_snapshots: int,
    sample_rate: float = 1000.0,
    amplitude: float = 1.0,
    initial_phase: float = 0.0
) -> np.ndarray:
    """Generate a discrete complex exponential baseband signal."""
    t = np.arange(n_snapshots) / sample_rate
    return amplitude * np.exp(1j * (2.0 * np.pi * frequency * t + initial_phase))


def generate_received_signal(
    array_config: ArrayConfig,
    target_angle: float,
    target_signal: np.ndarray,
    interference_sources: Optional[List[Tuple[float, np.ndarray]]] = None,
    noise_power: float = 0.01,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate the received signal snapshots X for the antenna array.

    x(t) = a(theta_s) * s(t) + sum_k a(theta_k) * i_k(t) + n(t)

    Args:
        array_config: ArrayConfig instance.
        target_angle: Angle of arrival for target spacecraft (degrees).
        target_signal: Complex 1D array of desired signal samples of length T.
        interference_sources: List of (angle_deg, interference_signal_array).
        noise_power: Total noise power sigma_n^2.
        random_seed: Optional seed for reproducibility.

    Returns:
        X: Complex snapshot matrix of shape (n_elements, n_snapshots).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_elements = array_config.n_elements
    n_snapshots = len(target_signal)
    X = np.zeros((n_elements, n_snapshots), dtype=complex)

    # 1. Target signal contribution
    a_target = steering_vector(
        n_elements,
        target_angle,
        array_config.wavelength,
        array_config.spacing
    )
    X += np.outer(a_target, target_signal)

    # 2. Interference contributions
    if interference_sources is not None:
        for angle_i, sig_i in interference_sources:
            a_i = steering_vector(
                n_elements,
                angle_i,
                array_config.wavelength,
                array_config.spacing
            )
            X += np.outer(a_i, sig_i)

    # 3. Additive complex Gaussian noise ~ CN(0, noise_power * I)
    sigma_per_dim = np.sqrt(noise_power / 2.0)
    noise = sigma_per_dim * (
        np.random.randn(n_elements, n_snapshots) + 1j * np.random.randn(n_elements, n_snapshots)
    )
    X += noise

    return X
