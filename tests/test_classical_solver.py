"""
Unit tests for classical MVDR solver and beam pattern calculation.
"""

import numpy as np
from eigenwall.signal_model import steering_vector
from eigenwall.beamforming import mvdr_weights, beam_pattern
from eigenwall.metrics import null_depth_db, residual_error


def test_mvdr_distortionless_constraint():
    N = 4
    theta_target = 20.0
    a_target = steering_vector(N, theta_target)

    # Well-conditioned synthetic covariance
    R = np.eye(N, dtype=complex) + 0.2 * np.outer(a_target, a_target.conj())

    w = mvdr_weights(R, a_target)

    # 1. Distortionless constraint: w^H * a_target == 1
    response = np.vdot(w, a_target)
    np.testing.assert_allclose(response.real, 1.0, atol=1e-10)
    np.testing.assert_allclose(response.imag, 0.0, atol=1e-10)

    # 2. Residual error: R * w proportional to a
    err = residual_error(R, a_target, w)
    assert err < 1e-10


def test_mvdr_nulling_behavior():
    N = 4
    theta_s = 20.0
    theta_i = -30.0

    a_s = steering_vector(N, theta_s)
    a_i = steering_vector(N, theta_i)

    # R with powerful interferer (power = 100) and small noise (power = 1)
    R = np.eye(N, dtype=complex) + 100.0 * np.outer(a_i, a_i.conj())

    w = mvdr_weights(R, a_s)

    # Null depth toward interference should be very deep (> 20 dB)
    nd = null_depth_db(w, theta_s, theta_i)
    assert nd > 20.0
