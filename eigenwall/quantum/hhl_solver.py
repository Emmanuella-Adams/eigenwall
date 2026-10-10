"""
HHL-style quantum linear system solver for small Hermitian matrices.
Implements State Prep -> QPE -> Controlled Eigenvalue Inversion Rotation -> Uncomputation.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate, RYGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from eigenwall.quantum.state_preparation import prepare_state
from eigenwall.quantum.qpe import compute_unitary_power


def build_hhl_circuit(
    matrix: np.ndarray,
    vector_b: np.ndarray,
    phase_qubits: int = 3,
    evolution_time: Optional[float] = None,
    scaling_constant: Optional[float] = None,
    measure_all: bool = False
) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    """
    Construct the complete HHL-style circuit:
    State Prep (|b>) -> QPE -> Controlled-Ry -> Inverse QPE (Uncomputation).

    Args:
        matrix: Hermitian positive definite matrix R (N, N).
        vector_b: RHS vector b (N,).
        phase_qubits: Number of clock/phase qubits m.
        evolution_time: Evolution parameter t. If None, sets t = pi / (max(eigvals)).
        scaling_constant: Constant C for rotation angle: sin(theta/2) = C / lambda_est.
                          Must satisfy C <= min(eigvals). If None, sets C = min(eigvals).
        measure_all: Whether to append classical measurement registers.

    Returns:
        circuit: QuantumCircuit
        meta: Dictionary of parameters used.
    """
    N = matrix.shape[0]
    n_sys = int(np.ceil(np.log2(N))) if N > 1 else 1
    m = phase_qubits

    # Matrix analysis
    eigvals = np.linalg.eigvalsh(matrix)
    lam_min = float(eigvals[0])
    lam_max = float(eigvals[-1])

    if evolution_time is None:
        t_val = np.pi / lam_max
    else:
        t_val = float(evolution_time)

    if scaling_constant is None:
        # Conservative C <= lambda_min
        c_val = 0.9 * lam_min
    else:
        c_val = float(scaling_constant)

    # Quantum registers
    # Ordering: ancilla (1), phase (m), sys (n_sys)
    ancilla_qr = QuantumRegister(1, name="ancilla")
    phase_qr = QuantumRegister(m, name="phase")
    sys_qr = QuantumRegister(n_sys, name="sys")

    if measure_all:
        c_anc = ClassicalRegister(1, name="c_anc")
        c_sys = ClassicalRegister(n_sys, name="c_sys")
        qc = QuantumCircuit(ancilla_qr, phase_qr, sys_qr, c_anc, c_sys)
    else:
        qc = QuantumCircuit(ancilla_qr, phase_qr, sys_qr)

    # 1. State Preparation |b> on sys
    prepare_state(qc, sys_qr, vector_b)

    # 2. QPE: Superposition on phase
    for k in range(m):
        qc.h(phase_qr[k])

    # 3. Controlled-U^(2^k) on sys
    dim = 2 ** n_sys
    if N < dim:
        mat_pad = np.zeros((dim, dim), dtype=complex)
        mat_pad[:N, :N] = matrix
        for i in range(N, dim):
            mat_pad[i, i] = 1.0
    else:
        mat_pad = matrix

    for k in range(m):
        power = 2 ** k
        u_pow = compute_unitary_power(mat_pad, t_val, power=power)
        u_gate = UnitaryGate(u_pow, label=f"U^{power}").control(1)
        qc.append(u_gate, [phase_qr[k]] + list(sys_qr))

    # 4. Inverse QFT on phase
    iqft = QFTGate(m).inverse()
    qc.append(iqft, phase_qr)

    # 5. Controlled Eigenvalue Inversion Rotation
    # For every computational state j in {1, ..., 2^m - 1} of phase register:
    # lambda_est = 2 * pi * j / (t * 2^m)
    # ratio = C / lambda_est
    # theta_j = 2 * arcsin(min(1.0, ratio))
    for j in range(1, 2 ** m):
        phi_j = j / (2 ** m)
        lam_est = 2.0 * np.pi * phi_j / t_val
        ratio = c_val / lam_est if lam_est > 1e-12 else 0.0
        ratio = min(1.0, max(0.0, ratio))
        theta_j = 2.0 * np.arcsin(ratio)

        # Multi-controlled rotation targeting ancilla conditioned on phase == j
        bitstring = format(j, f"0{m}b")
        # Flip qubits where bit is 0
        for bit_idx, bit_char in enumerate(reversed(bitstring)):
            if bit_char == '0':
                qc.x(phase_qr[bit_idx])

        # Controlled-Ry
        cry_gate = RYGate(theta_j).control(m, annotated=False)
        qc.append(cry_gate, list(phase_qr) + [ancilla_qr[0]])

        # Unflip qubits
        for bit_idx, bit_char in enumerate(reversed(bitstring)):
            if bit_char == '0':
                qc.x(phase_qr[bit_idx])

    # 6. Uncompute QPE
    # 6a. Forward QFT on phase
    qft = QFTGate(m)
    qc.append(qft, phase_qr)

    # 6b. Controlled-U^(-2^k) on sys
    for k in reversed(range(m)):
        power = 2 ** k
        u_pow_inv = compute_unitary_power(mat_pad, t_val, power=-power)
        u_inv_gate = UnitaryGate(u_pow_inv, label=f"U^-{power}").control(1)
        qc.append(u_inv_gate, [phase_qr[k]] + list(sys_qr))

    # 6c. Superposition uncompute
    for k in range(m):
        qc.h(phase_qr[k])

    # 7. Optional measurements
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
        "kappa": lam_max / lam_min,
    }
    return qc, meta


def quantum_linear_solver(
    matrix: np.ndarray,
    vector_b: np.ndarray,
    phase_qubits: int = 4,
    evolution_time: Optional[float] = None,
    scaling_constant: Optional[float] = None,
    shots: int = 8192,
    use_statevector: bool = True
) -> Dict[str, Any]:
    """
    Execute the quantum linear system solver for matrix * x = b.

    Returns:
        result: Dictionary containing:
            - circuit: The built QuantumCircuit
            - solution_state: The post-selected normalized complex statevector |x_q>
            - success_probability: Probability of measuring ancilla = 1
            - classical_solution: Exact classical solution x_c = R^-1 * b
            - classical_state: Normalized classical solution |x_c>
            - fidelity: State fidelity |<x_c|x_q>|^2
            - metadata: Simulation metadata
    """
    N = matrix.shape[0]
    n_sys = int(np.ceil(np.log2(N))) if N > 1 else 1
    m = phase_qubits

    # Exact classical solution for benchmarking
    classical_x = np.linalg.solve(matrix, vector_b)
    classical_norm = classical_x / np.linalg.norm(classical_x)

    # 1. Build circuit (without terminal measurements for statevector inspection)
    qc, meta = build_hhl_circuit(
        matrix=matrix,
        vector_b=vector_b,
        phase_qubits=m,
        evolution_time=evolution_time,
        scaling_constant=scaling_constant,
        measure_all=False
    )

    # 2. Statevector simulation to extract post-selected state
    sv = Statevector.from_instruction(qc)
    sv_data = sv.data  # shape (2^(1 + m + n_sys),)

    # In Qiskit, qubit ordering in state index is:
    # qubit 0: ancilla
    # qubits 1 to m: phase
    # qubits m+1 to m+n_sys: sys
    total_qubits = 1 + m + n_sys
    solution_amplitudes = np.zeros(2 ** n_sys, dtype=complex)
    ancilla_1_prob = 0.0

    for idx, amp in enumerate(sv_data):
        prob = np.abs(amp) ** 2
        # Check ancilla bit (bit 0)
        ancilla_bit = idx & 1
        if ancilla_bit == 1:
            ancilla_1_prob += prob
            # Phase bits: bits 1 to m
            phase_val = (idx >> 1) & ((1 << m) - 1)
            # System bits: bits m+1 to m+n_sys
            sys_val = (idx >> (1 + m)) & ((1 << n_sys) - 1)
            # Ideally after uncomputation, phase is 0
            if phase_val == 0:
                solution_amplitudes[sys_val] += amp

    # Post-selected normalized state on system register
    q_norm = np.linalg.norm(solution_amplitudes[:N])
    if q_norm > 1e-12:
        quantum_solution_state = solution_amplitudes[:N] / q_norm
    else:
        # Fallback if uncomputed phase had numerical residue: project all ancilla=1
        fallback_amps = np.zeros(N, dtype=complex)
        for idx, amp in enumerate(sv_data):
            if (idx & 1) == 1:
                sys_val = (idx >> (1 + m)) & ((1 << n_sys) - 1)
                if sys_val < N:
                    fallback_amps[sys_val] += amp
        fn = np.linalg.norm(fallback_amps)
        quantum_solution_state = fallback_amps / fn if fn > 1e-12 else np.ones(N) / np.sqrt(N)

    # Compute fidelity with classical normalized solution
    fid = float(np.abs(np.vdot(classical_norm, quantum_solution_state)) ** 2)

    # 3. Shot-based execution for physical verification
    qc_meas, _ = build_hhl_circuit(
        matrix=matrix,
        vector_b=vector_b,
        phase_qubits=m,
        evolution_time=evolution_time,
        scaling_constant=scaling_constant,
        measure_all=True
    )
    sim = AerSimulator()
    transpiled = transpile(qc_meas, sim)
    job = sim.run(transpiled, shots=shots)
    counts = job.result().get_counts()

    return {
        "circuit": qc,
        "measured_circuit": qc_meas,
        "counts": counts,
        "solution_state": quantum_solution_state,
        "success_probability": float(ancilla_1_prob),
        "classical_solution": classical_x,
        "classical_state": classical_norm,
        "fidelity": fid,
        "metadata": meta,
    }
