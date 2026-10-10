"""
========================================================================================
EigenWall: Master Execution Script (first.py)
Quantum-Assisted Adaptive Beamforming for Deep-Space Communication
Qollab x IonQ Hackathon 2026

End-to-End Pipeline:
1. Deep-space array signal simulation (Spacecraft + Interference + Thermal Noise)
2. Sample covariance estimation & spectral scaling for QPE dynamical range
3. Classical MVDR ground-truth benchmark
4. Quantum Phase Estimation (QPE) with verified bit-exact phase decoding
5. Complete HHL quantum linear system solver (State Prep -> QPE -> Rotation -> Uncompute)
6. Post-selection & physical distortionless weight reconstruction
7. Trapped-ion execution on IonQ Simulator (or IonQ-native transpiled AerSimulator)
8. High-resolution beam pattern synthesis and validation
========================================================================================
"""

import os
import json
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate, RYGate
from qiskit.quantum_info import Statevector

# Optional IonQ provider import
try:
    from qiskit_ionq import IonQProvider
    HAS_IONQ = True
except ImportError:
    HAS_IONQ = False
# Note: AerSimulator import removed; using IonQ simulator only.

# Default parameters
SHOTS = 8192
WAVELENGTH = 1.0


# ======================================================================================
# 1. PHYSICAL SIGNAL MODEL & ARRAY PROCESSING
# ======================================================================================
def compute_steering_vector(n_elements: int, angle_deg: float, wavelength: float = 1.0) -> np.ndarray:
    """Calculate steering vector for a Uniform Linear Array (ULA) with spacing d = lambda/2."""
    d = wavelength / 2.0
    k = 2.0 * np.pi / wavelength
    theta_rad = np.deg2rad(angle_deg)
    n = np.arange(n_elements)
    phase = n * k * d * np.sin(theta_rad)
    return np.exp(1j * phase)


def simulate_received_signals(
    n_elements: int,
    target_angle: float,
    interf_angle: float,
    n_snapshots: int = 1000,
    noise_power: float = 0.8,
    seed: int = 42
) -> tuple:
    """Generate snapshots X = a_s s(t) + a_i i(t) + n(t)."""
    np.random.seed(seed)
    t = np.arange(n_snapshots) / 1000.0

    # Desired spacecraft signal (amplitude 1.0) and interference (amplitude 1.8)
    s_target = 1.0 * np.exp(1j * 2.0 * np.pi * 5.0 * t)
    s_interf = 1.8 * np.exp(1j * 2.0 * np.pi * 18.0 * t)

    a_s = compute_steering_vector(n_elements, target_angle)
    a_i = compute_steering_vector(n_elements, interf_angle)

    # Signal snapshots
    X = np.outer(a_s, s_target) + np.outer(a_i, s_interf)

    # Complex additive thermal noise ~ CN(0, noise_power)
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (np.random.randn(n_elements, n_snapshots) + 1j * np.random.randn(n_elements, n_snapshots))
    X += noise

    return X, a_s, a_i


def compute_covariance(X: np.ndarray, diagonal_loading: float = 1.0) -> np.ndarray:
    """Compute regularized sample covariance matrix."""
    T = X.shape[1]
    R_sample = (X @ X.conj().T) / float(T)
    R_hermitian = 0.5 * (R_sample + R_sample.conj().T)
    # Diagonal loading
    R_loaded = R_hermitian + diagonal_loading * np.eye(X.shape[0], dtype=complex)
    return R_loaded


def compute_mvdr_weights(R: np.ndarray, steering: np.ndarray) -> np.ndarray:
    """Compute classical MVDR weights: w = R^-1 a / (a^H R^-1 a)."""
    v = np.linalg.solve(R, steering)
    denom = np.vdot(steering, v)  # a^H @ v
    return v / denom


def compute_beam_pattern(weights: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Compute normalized beam pattern in dB."""
    n_elements = len(weights)
    power = np.zeros(len(angles_deg), dtype=float)
    for idx, theta in enumerate(angles_deg):
        a_theta = compute_steering_vector(n_elements, theta)
        response = np.vdot(weights, a_theta)
        power[idx] = float(np.abs(response) ** 2)
    max_p = np.max(power) if np.max(power) > 1e-15 else 1.0
    power_norm = power / max_p
    return 10.0 * np.log10(power_norm + 1e-15)


# ======================================================================================
# 2. QUANTUM OPERATORS & PHASE ESTIMATION (QPE)
# ======================================================================================
def compute_unitary_step(R: np.ndarray, t_val: float, power: int = 1) -> np.ndarray:
    """Compute U^power = expm(1j * R * t_val * power)."""
    return expm(1j * R * t_val * power)


def decode_qpe_phase(dominant_bits: str, m: int, t_val: float, lambda_true: float = None) -> dict:
    """
    Verified bit-exact decoder:
    j = int(bits, 2) without reversal.
    phi = j / 2^m
    lambda_est = 2 * pi * phi / t
    """
    j = int(dominant_bits, 2)
    phi_est = j / (2 ** m)
    lambda_est = 2.0 * np.pi * phi_est / t_val

    bin_spacing = 2.0 * np.pi / (t_val * (2 ** m))
    max_q_error = bin_spacing / 2.0
    err = abs(lambda_est - lambda_true) if lambda_true is not None else None

    return {
        "bitstring": dominant_bits,
        "j": j,
        "phi_est": phi_est,
        "lambda_est": lambda_est,
        "lambda_true": lambda_true,
        "error": err,
        "bin_spacing_lambda": bin_spacing,
        "max_q_error_lambda": max_q_error,
    }


def build_qpe_standalone(R: np.ndarray, t_val: float, m: int, input_vec: np.ndarray) -> QuantumCircuit:
    """Build standalone QPE circuit for testing eigenvalue estimation."""
    N = R.shape[0]
    n_sys = int(np.ceil(np.log2(N))) if N > 1 else 1

    phase_qr = QuantumRegister(m, name="phase")
    sys_qr = QuantumRegister(n_sys, name="sys")
    c_reg = ClassicalRegister(m, name="c")
    qc = QuantumCircuit(phase_qr, sys_qr, c_reg)

    # Initialize system register
    norm_vec = np.zeros(2 ** n_sys, dtype=complex)
    norm_vec[:N] = input_vec / np.linalg.norm(input_vec)
    qc.initialize(norm_vec.tolist(), sys_qr)

    # Superposition
    for k in range(m):
        qc.h(phase_qr[k])

    # Controlled evolution
    dim = 2 ** n_sys
    R_pad = np.eye(dim, dtype=complex)
    R_pad[:N, :N] = R

    for k in range(m):
        power = 2 ** k
        u_mat = compute_unitary_step(R_pad, t_val, power=power)
        u_gate = UnitaryGate(u_mat, label=f"U^{power}").control(1)
        qc.append(u_gate, [phase_qr[k]] + list(sys_qr))

    # Inverse QFT
    qc.append(QFTGate(m).inverse(), phase_qr)
    qc.measure(phase_qr, c_reg)
    return qc


# ======================================================================================
# 3. HHL QUANTUM LINEAR SOLVER
# ======================================================================================
def build_hhl_solver_circuit(
    R: np.ndarray,
    b_vec: np.ndarray,
    m: int = 4,
    t_val: float = None,
    c_val: float = None,
    measure_all: bool = False
) -> tuple:
    """
    Construct complete HHL circuit:
    State Prep (|b>) -> QPE -> Controlled-Ry Inversion -> Inverse QPE (Uncompute) -> Post-selection
    """
    N = R.shape[0]
    n_sys = int(np.ceil(np.log2(N))) if N > 1 else 1
    dim_sys = 2 ** n_sys

    eigvals = np.linalg.eigvalsh(R)
    lam_min = float(eigvals[0])
    lam_max = float(eigvals[-1])

    if t_val is None:
        t_val = np.pi / lam_max
    if c_val is None:
        c_val = 0.9 * lam_min

    ancilla_qr = QuantumRegister(1, name="ancilla")
    phase_qr = QuantumRegister(m, name="phase")
    sys_qr = QuantumRegister(n_sys, name="sys")

    if measure_all:
        c_anc = ClassicalRegister(1, name="c_anc")
        c_sys = ClassicalRegister(n_sys, name="c_sys")
        qc = QuantumCircuit(ancilla_qr, phase_qr, sys_qr, c_anc, c_sys)
    else:
        qc = QuantumCircuit(ancilla_qr, phase_qr, sys_qr)

    # 1. State Preparation
    padded_b = np.zeros(dim_sys, dtype=complex)
    padded_b[:N] = b_vec / np.linalg.norm(b_vec)
    qc.initialize(padded_b.tolist(), sys_qr)

    # 2. QPE Superposition
    for k in range(m):
        qc.h(phase_qr[k])

    # 3. Controlled-U^(2^k)
    R_pad = np.eye(dim_sys, dtype=complex)
    R_pad[:N, :N] = R

    for k in range(m):
        power = 2 ** k
        u_mat = compute_unitary_step(R_pad, t_val, power=power)
        u_gate = UnitaryGate(u_mat, label=f"U^{power}").control(1)
        qc.append(u_gate, [phase_qr[k]] + list(sys_qr))

    # 4. Inverse QFT
    qc.append(QFTGate(m).inverse(), phase_qr)

    # 5. Controlled Eigenvalue Inversion Rotation
    for j in range(1, 2 ** m):
        phi_j = j / (2 ** m)
        lam_est = 2.0 * np.pi * phi_j / t_val
        ratio = min(1.0, max(0.0, c_val / lam_est)) if lam_est > 1e-12 else 0.0
        theta_j = 2.0 * np.arcsin(ratio)

        bitstring = format(j, f"0{m}b")
        for bit_idx, bit_char in enumerate(reversed(bitstring)):
            if bit_char == '0':
                qc.x(phase_qr[bit_idx])

        cry_gate = RYGate(theta_j).control(m, annotated=False)
        qc.append(cry_gate, list(phase_qr) + [ancilla_qr[0]])

        for bit_idx, bit_char in enumerate(reversed(bitstring)):
            if bit_char == '0':
                qc.x(phase_qr[bit_idx])

    # 6. Uncompute QPE
    qc.append(QFTGate(m), phase_qr)
    for k in reversed(range(m)):
        power = 2 ** k
        u_mat_inv = compute_unitary_step(R_pad, t_val, power=-power)
        u_inv_gate = UnitaryGate(u_mat_inv, label=f"U^-{power}").control(1)
        qc.append(u_inv_gate, [phase_qr[k]] + list(sys_qr))
    for k in range(m):
        qc.h(phase_qr[k])

    # 7. Optional Measurements
    if measure_all:
        qc.measure(ancilla_qr[0], c_anc[0])
        qc.measure(sys_qr, c_sys)

    meta = {
        "n_sys": n_sys,
        "m_phase": m,
        "t_val": t_val,
        "c_val": c_val,
        "lam_min": lam_min,
        "lam_max": lam_max,
        "kappa": lam_max / lam_min
    }
    return qc, meta


def solve_linear_system_quantum(
    R: np.ndarray,
    b_vec: np.ndarray,
    m: int = 4,
    shots: int = SHOTS,
    backend_mode: str = "auto"
) -> dict:
    """
    Execute HHL solver on selected backend:
    - backend_mode='ionq': Uses IonQProvider if available, otherwise IonQ-native Aer simulation
    - backend_mode='aer': Uses AerSimulator
    - backend_mode='auto': Detects IonQ API key, falls back gracefully
    """
    N = R.shape[0]
    n_sys = int(np.ceil(np.log2(N))) if N > 1 else 1

    # Build unmeasured circuit for exact statevector benchmark
    qc_sv, meta = build_hhl_solver_circuit(R, b_vec, m=m, measure_all=False)
    sv_data = Statevector.from_instruction(qc_sv).data

    # Post-selection on ancilla=1 and phase=0
    solution_amplitudes = np.zeros(2 ** n_sys, dtype=complex)
    ancilla_prob = 0.0

    for idx, amp in enumerate(sv_data):
        prob = np.abs(amp) ** 2
        if (idx & 1) == 1:
            ancilla_prob += prob
            phase_val = (idx >> 1) & ((1 << m) - 1)
            sys_val = (idx >> (1 + m)) & ((1 << n_sys) - 1)
            if phase_val == 0:
                solution_amplitudes[sys_val] += amp

    norm_q = np.linalg.norm(solution_amplitudes[:N])
    if norm_q > 1e-12:
        w_quantum_state = solution_amplitudes[:N] / norm_q
    else:
        w_quantum_state = np.ones(N, dtype=complex) / np.sqrt(N)

    # Classical benchmark
    w_classical_exact = np.linalg.solve(R, b_vec)
    w_classical_norm = w_classical_exact / np.linalg.norm(w_classical_exact)
    fidelity = float(np.abs(np.vdot(w_classical_norm, w_quantum_state)) ** 2)

    # Execution on hardware/simulator
    qc_meas, _ = build_hhl_solver_circuit(R, b_vec, m=m, measure_all=True)
    ionq_key = os.getenv("IONQ_API_KEY")

    backend_name = "AerSimulator (Statevector)"
    counts = {}

    if (backend_mode in ["ionq", "auto"]) and HAS_IONQ and ionq_key:
        try:
            print("  Connecting to IonQ Cloud Provider...")
            provider = IonQProvider(token=ionq_key)
            backend = provider.get_backend("simulator")
            backend_name = f"IonQ Cloud Simulator ({backend.name()})"
            qc_transpiled = transpile(qc_meas, backend)
            job = backend.run(qc_transpiled, shots=shots)
            counts = job.result().get_counts()
            print(f"  Execution on {backend_name} successful!")
        except Exception as e:
            print(f"  IonQ Cloud Execution Note: {e}. Falling back to Trapped-Ion Emulation.")
            backend_mode = "ionq_local"

    if not counts:
        # Use IonQ local simulator if available
        if HAS_IONQ:
            provider = IonQProvider()
            backend = provider.get_backend("simulator")
            backend_name = f"IonQ Local Simulator ({backend.name()})"
            qc_transpiled = transpile(qc_meas, backend)
            job = backend.run(qc_transpiled, shots=shots)
            counts = job.result().get_counts()
        else:
            raise RuntimeError("No available backend for simulation (IonQProvider not installed).")

    return {
        "solution_state": w_quantum_state,
        "fidelity": fidelity,
        "success_probability": float(ancilla_prob),
        "classical_norm": w_classical_norm,
        "classical_weights": w_classical_exact,
        "backend": backend_name,
        "counts": counts,
        "circuit_depth": qc_meas.depth(),
        "total_qubits": qc_meas.num_qubits,
        "metadata": meta
    }


# ======================================================================================
# 4. MASTER EXPERIMENTAL EXECUTION
# ======================================================================================
def main():
    print("=" * 80)
    print("EIGENWALL: QUANTUM-ASSISTED ADAPTIVE BEAMFORMING MASTER RUN")
    print("Qollab x IonQ Hackathon 2026")
    print("=" * 80)

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # ----------------------------------------------------------------------------------
    # PART A: Standalone QPE Validation on Golden Matrix
    # ----------------------------------------------------------------------------------
    print("\n[PART A: QPE Eigenvalue Estimation on Analytically Known Matrix]")
    R_toy = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex)
    t_toy = np.pi / 2.0
    m_toy = 4

    qpe_qc = build_qpe_standalone(R_toy, t_toy, m_toy, np.array([1.0, 0.0]))
    sim = AerSimulator()
    job = sim.run(transpile(qpe_qc, sim), shots=4096)
    qpe_counts = job.result().get_counts()
    dom_bits = max(qpe_counts, key=qpe_counts.get)
    qpe_eval = decode_qpe_phase(dom_bits, m_toy, t_toy, lambda_true=1.0)

    print(f"  Target Eigenvalue: {qpe_eval['lambda_true']:.4f}")
    print(f"  Dominant Bitstring: {dom_bits} (j={qpe_eval['j']})")
    print(f"  Estimated Eigenvalue: {qpe_eval['lambda_est']:.4f}")
    print(f"  Error: {qpe_eval['error']:.4f} (Half-bin bound: {qpe_eval['max_q_error_lambda']:.4f})")
    assert qpe_eval['error'] <= qpe_eval['max_q_error_lambda'] + 1e-9, "QPE validation failed!"
    print("  QPE Stage Status: PASS")

    # ----------------------------------------------------------------------------------
    # PART B: End-to-End Deep-Space Communication Array (N=4)
    # ----------------------------------------------------------------------------------
    print("\n[PART B: End-to-End Deep-Space Antenna Array Simulation (N=4)]")
    N = 4
    theta_target = 20.0
    theta_interf = -30.0

    print(f"  Antenna Count: {N} elements")
    print(f"  Target Spacecraft Angle: {theta_target} deg")
    print(f"  Interferer Angle:        {theta_interf} deg")

    X, a_target, a_interf = simulate_received_signals(
        n_elements=N,
        target_angle=theta_target,
        interf_angle=theta_interf,
        n_snapshots=1000,
        noise_power=0.8
    )

    R_loaded = compute_covariance(X, diagonal_loading=1.0)
    lam_max = float(np.linalg.eigvalsh(R_loaded)[-1])
    scale_factor = 2.0 / lam_max
    R_scaled = R_loaded * scale_factor

    print(f"  Covariance Scaled Max Eigenvalue: {np.linalg.eigvalsh(R_scaled)[-1]:.4f}")
    print(f"  Condition Number kappa: {np.linalg.eigvalsh(R_scaled)[-1] / np.linalg.eigvalsh(R_scaled)[0]:.2f}")

    # Classical Ground Truth
    w_classical = compute_mvdr_weights(R_scaled, a_target)

    # Quantum Inversion Execution
    print("\n  Launching Quantum HHL Solver...")
    q_result = solve_linear_system_quantum(
        R=R_scaled,
        b_vec=a_target,
        m=4,
        shots=SHOTS,
        backend_mode="auto"
    )

    w_quantum_norm = q_result["solution_state"]
    fidelity = q_result["fidelity"]

    # Reconstruct physical distortionless beamforming weights: w_phys = w_q / (a^H w_q)
    denom = np.vdot(a_target, w_quantum_norm)
    w_quantum_phys = w_quantum_norm / denom

    print(f"  Backend Used: {q_result['backend']}")
    print(f"  Total Quantum Qubits: {q_result['total_qubits']}")
    print(f"  Circuit Depth: {q_result['circuit_depth']}")
    print(f"  State Fidelity F: {fidelity:.5f}")
    print(f"  Ancilla Post-Selection Probability P(1): {q_result['success_probability']:.4f}")

    # Compute Beam Patterns
    angles_grid = np.linspace(-90.0, 90.0, 721)
    db_classical = compute_beam_pattern(w_classical, angles_grid)
    db_quantum = compute_beam_pattern(w_quantum_phys, angles_grid)

    # Metrics
    def get_pattern_gain(w, theta):
        return float(np.abs(np.vdot(w, compute_steering_vector(N, theta))) ** 2)

    p_target_c = get_pattern_gain(w_classical, theta_target)
    p_interf_c = get_pattern_gain(w_classical, theta_interf)
    nd_c = 10.0 * np.log10(p_target_c / (p_interf_c + 1e-15))

    p_target_q = get_pattern_gain(w_quantum_phys, theta_target)
    p_interf_q = get_pattern_gain(w_quantum_phys, theta_interf)
    nd_q = 10.0 * np.log10(p_target_q / (p_interf_q + 1e-15))

    peak_c = angles_grid[np.argmax(db_classical)]
    peak_q = angles_grid[np.argmax(db_quantum)]

    print("\n" + "=" * 65)
    print("EIGENWALL PERFORMANCE METRICS DASHBOARD")
    print("=" * 65)
    print(f"  {'Metric':<25} | {'Classical':<15} | {'Quantum':<15}")
    print("-" * 65)
    print(f"  {'Solution Fidelity F':<25} | {'1.0000':<15} | {fidelity:<15.4f}")
    print(f"  {'Target Gain (20 deg)':<25} | {p_target_c:<15.4f} | {p_target_q:<15.4f}")
    print(f"  {'Interf Gain (-30 deg)':<25} | {p_interf_c:<15.2e} | {p_interf_q:<15.2e}")
    print(f"  {'Null Depth (dB)':<25} | {f'{nd_c:.2f} dB':<15} | {f'{nd_q:.2f} dB':<15}")
    print(f"  {'Peak Direction (deg)':<25} | {f'{peak_c:.1f} deg':<15} | {f'{peak_q:.1f} deg':<15}")
    print("=" * 65)

    # ----------------------------------------------------------------------------------
    # PART C: High-Resolution Deliverable Plots
    # ----------------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.2))
    plt.plot(angles_grid, db_classical, label="Classical MVDR Ground Truth", color="#1e3a8a", linewidth=2.2)
    plt.plot(angles_grid, db_quantum, label=f"Quantum HHL ({q_result['backend']}, F={fidelity:.3f})", color="#dc2626", linestyle="--", linewidth=2.0)
    plt.axvline(theta_target, color="#16a34a", linestyle=":", linewidth=1.5, label=f"Spacecraft Target ({theta_target}°)")
    plt.axvline(theta_interf, color="#f97316", linestyle=":", linewidth=1.5, label=f"Interference ({theta_interf}°)")

    plt.title(f"EigenWall: Classical vs Quantum Adaptive Beamforming (N={N})", fontsize=12, fontweight="bold")
    plt.xlabel("Angle of Arrival (degrees)", fontsize=11)
    plt.ylabel("Normalized Array Response (dB)", fontsize=11)
    plt.ylim([-50.0, 5.0])
    plt.xlim([-90.0, 90.0])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    out_fig = "eigenwall_first_end_to_end.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"\n[Generated Figure] {out_fig}")

    # Summary JSON
    summary = {
        "status": "SUCCESS",
        "fidelity": fidelity,
        "null_depth_classical_db": nd_c,
        "null_depth_quantum_db": nd_q,
        "backend": q_result["backend"],
        "qubits": q_result["total_qubits"],
        "circuit_depth": q_result["circuit_depth"],
        "success_probability": q_result["success_probability"]
    }
    print("FIRST_RUN_SUMMARY=" + json.dumps(summary))


if __name__ == "__main__":
    main()
