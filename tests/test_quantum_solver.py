"""
Unit tests for quantum subroutines: QPE and HHL-style linear system solver.
"""

import numpy as np
from eigenwall.quantum.qpe import run_qpe
from eigenwall.quantum.hhl_solver import quantum_linear_solver
from eigenwall.metrics import state_fidelity


def test_qpe_golden_matrix():
    # R1 = diag(1, 2)
    R1 = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex)
    t_val = np.pi / 2.0  # lambda_max = 2.0 -> phi_max = 0.5
    m = 4

    # Test eigenvector 0: [1, 0] with lambda = 1.0
    _, res0 = run_qpe(R1, t_val, m, input_state_vector=np.array([1.0, 0.0]), lambda_true=1.0, shots=2048)
    assert res0["error"] <= res0["max_q_error_lambda"] + 1e-9
    assert np.isclose(res0["lambda_est"], 1.0, atol=0.01)

    # Test eigenvector 1: [0, 1] with lambda = 2.0
    _, res1 = run_qpe(R1, t_val, m, input_state_vector=np.array([0.0, 1.0]), lambda_true=2.0, shots=2048)
    assert res1["error"] <= res1["max_q_error_lambda"] + 1e-9
    assert np.isclose(res1["lambda_est"], 2.0, atol=0.01)


def test_hhl_golden_test_a():
    """Golden Test A: A = diag(1, 2), b = [1, 1]^T."""
    A = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex)
    b = np.array([1.0, 1.0], dtype=complex)

    # Analytical solution: x = [1, 0.5]^T
    expected_x = np.array([1.0, 0.5], dtype=complex)
    expected_state = expected_x / np.linalg.norm(expected_x)

    res = quantum_linear_solver(
        matrix=A,
        vector_b=b,
        phase_qubits=4,
        evolution_time=np.pi / 2.0,
        scaling_constant=0.8,
        shots=4096
    )

    fid = state_fidelity(expected_state, res["solution_state"])
    print(f"Golden Test A Fidelity: {fid:.5f}")
    assert fid > 0.98
    assert res["success_probability"] > 0.05


def test_hhl_golden_test_b():
    """Golden Test B: A = [[1, 0.2], [0.2, 2]], b = [1, 1]^T."""
    A = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=complex)
    b = np.array([1.0, 1.0], dtype=complex)

    res = quantum_linear_solver(
        matrix=A,
        vector_b=b,
        phase_qubits=4,
        shots=4096
    )

    fid = res["fidelity"]
    print(f"Golden Test B Fidelity: {fid:.5f}")
    assert fid > 0.95
