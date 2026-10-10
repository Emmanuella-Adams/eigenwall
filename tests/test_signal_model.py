"""
Unit tests for antenna array signal modeling.
"""

import numpy as np
from eigenwall.signal_model import ArrayConfig, steering_vector, generate_received_signal, generate_sinusoid


def test_steering_vector_properties():
    N = 4
    angle = 30.0
    a = steering_vector(N, angle)

    # 1. Dimension check
    assert len(a) == N
    assert a.dtype == complex

    # 2. Magnitude check (every element has magnitude 1)
    magnitudes = np.abs(a)
    np.testing.assert_allclose(magnitudes, np.ones(N), atol=1e-12)

    # 3. Broadside (0 deg) steering vector should be all ones
    a_broadside = steering_vector(N, 0.0)
    np.testing.assert_allclose(a_broadside, np.ones(N), atol=1e-12)

    # 4. Phase progression check
    expected_phase_step = 2.0 * np.pi * 0.5 * np.sin(np.deg2rad(angle))
    phase_diffs = np.diff(np.unwrap(np.angle(a)))
    np.testing.assert_allclose(phase_diffs, np.full(N - 1, expected_phase_step), atol=1e-10)


def test_received_signal_generation():
    config = ArrayConfig(n_elements=4, wavelength=1.0)
    T = 500
    target_signal = generate_sinusoid(10.0, T)
    interf_signal = generate_sinusoid(25.0, T, amplitude=5.0)

    X = generate_received_signal(
        array_config=config,
        target_angle=20.0,
        target_signal=target_signal,
        interference_sources=[(-30.0, interf_signal)],
        noise_power=0.01,
        random_seed=42
    )

    assert X.shape == (4, T)
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()
