"""
Quantum Phase Estimation (QPE) module for estimating eigenvalues of Hermitian operators.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit_aer import AerSimulator
from eigenwall.quantum.state_preparation import prepare_state


def compute_unitary_power(R: np.ndarray, t_val: float, power: int = 1) -> np.ndarray:
    """Compute U^power = expm(1j * R * t_val * power)."""
    U = expm(1j * R * t_val * power)
    err = np.linalg.norm(U.conj().T @ U - np.eye(len(R)))
    if err > 1e-8:
        raise ValueError(f"Matrix exponential is non-unitary: error = {err:.2e}")
    return U


def decode_phase(
    dominant_bits: str,
    m: int,
    t_val: float,
    lambda_true: Optional[float] = None
) -> Dict[str, Any]:
    """
    Convert a Qiskit QPE measurement bitstring to an eigenvalue estimate.
    Correct bitstring interpretation: j = int(dominant_bits, 2) without reversal.
    """
    j = int(dominant_bits, 2)
    phi_est = j / (2 ** m)
    lambda_est = 2.0 * np.pi * phi_est / t_val

    bin_spacing_phi = 1.0 / (2 ** m)
    bin_spacing_lambda = 2.0 * np.pi / (t_val * (2 ** m))
    max_q_error_lambda = bin_spacing_lambda / 2.0

    error = abs(lambda_est - lambda_true) if lambda_true is not None else None

    return {
        "bitstring": dominant_bits,
        "j": j,
        "phi_est": phi_est,
        "lambda_est": lambda_est,
        "lambda_true": lambda_true,
        "error": error,
        "bin_spacing_phi": bin_spacing_phi,
        "bin_spacing_lambda": bin_spacing_lambda,
        "max_q_error_lambda": max_q_error_lambda,
    }


def build_qpe_circuit(
    R: np.ndarray,
    t_val: float,
    m: int,
    input_state_vector: Optional[np.ndarray] = None,
    measure: bool = True
) -> QuantumCircuit:
    """
    Build a Quantum Phase Estimation circuit for unitary U = expm(i * R * t).

    Args:
        R: Hermitian matrix (N, N).
        t_val: Evolution parameter ensuring lambda_max * t / (2*pi) < 1.
        m: Number of phase evaluation qubits.
        input_state_vector: Optional input state on system register (length N).
        measure: If True, attaches classical register and measures phase qubits.

    Returns:
        QuantumCircuit: QPE circuit.
    """
    N = R.shape[0]
    n_sys = int(np.ceil(np.log2(N))) if N > 1 else 1

    phase_qr = QuantumRegister(m, name="phase")
    sys_qr = QuantumRegister(n_sys, name="sys")
    if measure:
        c_reg = ClassicalRegister(m, name="c")
        qc = QuantumCircuit(phase_qr, sys_qr, c_reg)
    else:
        qc = QuantumCircuit(phase_qr, sys_qr)

    # 1. State preparation on system register
    if input_state_vector is not None:
        prepare_state(qc, sys_qr, input_state_vector)

    # 2. Equal superposition on phase register
    for k in range(m):
        qc.h(phase_qr[k])

    # 3. Controlled-U^(2^k) operations
    # If N is not power of 2, pad R
    dim = 2 ** n_sys
    if N < dim:
        R_pad = np.zeros((dim, dim), dtype=complex)
        R_pad[:N, :N] = R
        # Set unused diagonal entries to 1 to preserve well-behaved spectrum
        for i in range(N, dim):
            R_pad[i, i] = 1.0
    else:
        R_pad = R

    for k in range(m):
        power = 2 ** k
        U_pow = compute_unitary_power(R_pad, t_val, power=power)
        u_gate = UnitaryGate(U_pow, label=f"U^{power}").control(1)
        qc.append(u_gate, [phase_qr[k]] + list(sys_qr))

    # 4. Inverse Quantum Fourier Transform (IQFT) on phase register
    iqft = QFTGate(m).inverse()
    qc.append(iqft, phase_qr)

    # 5. Measure phase qubits
    if measure:
        qc.measure(phase_qr, c_reg)

    return qc


def run_qpe(
    R: np.ndarray,
    t_val: float,
    m: int,
    input_state_vector: np.ndarray,
    lambda_true: Optional[float] = None,
    shots: int = 8192
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Execute QPE on AerSimulator, return counts and parsed result dictionary.
    """
    qc = build_qpe_circuit(R, t_val, m, input_state_vector=input_state_vector, measure=True)
    sim = AerSimulator(method="statevector")
    qc_t = transpile(qc, sim)
    job = sim.run(qc_t, shots=shots)
    counts = job.result().get_counts()

    dominant_bits = max(counts, key=counts.get)
    result = decode_phase(dominant_bits, m, t_val, lambda_true=lambda_true)
    result["counts"] = counts
    result["dominant_prob"] = counts[dominant_bits] / shots
    return counts, result
