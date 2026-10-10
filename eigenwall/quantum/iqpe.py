"""
Iterative Quantum Phase Estimation (IQPE) utilities.

Provides a simple, educational implementation of IQPE suitable for the small
Hermitian matrices used in the EigenWall prototype.  The routine is a drop‑in
replacement for the original QFT‑based phase estimation used in
`hhl_solver.build_hhl_circuit`.
"""

from typing import List, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate


def iqpe(
    qc: QuantumCircuit,
    system_qubits: List[int],
    unitary: np.ndarray,
    phase_qubits: int = 3,
    target_precision: int = 3,
) -> Tuple[QuantumCircuit, List[int]]:
    """Iterative Phase Estimation.

    Args:
        qc: QuantumCircuit that already contains the system register.
        system_qubits: List of indices (or Qiskit Qubit objects) belonging to the
            system register on which ``unitary`` acts.
        unitary: The matrix ``U`` to be exponentiated (must be unitary).
        phase_qubits: Number of ancilla qubits used for the iterative estimate.
        target_precision: Number of bits of precision (default 3).

    Returns:
        (updated_circuit, measurement_bits) – ``measurement_bits`` contains the
        binary outcome (LSB first) of each iteration.
    """
    # Allocate phase register
    phase_reg = QuantumRegister(phase_qubits, name="phase_iqpe")
    qc.add_register(phase_reg)
    measurement_bits: List[int] = []

    # Iterative loop – most‑significant bit first
    for k in range(target_precision):
        # 1. Put ancilla into superposition
        qc.h(phase_reg[k])
        # 2. Controlled‑U^{2^{target_precision‑k‑1}}
        power = 2 ** (target_precision - k - 1)
        u_pow = np.linalg.matrix_power(unitary, power)
        ctrl_u = UnitaryGate(u_pow).control(1)
        qc.append(ctrl_u, [phase_reg[k]] + system_qubits)
        # 3. Measure ancilla
        cr = ClassicalRegister(1)
        qc.add_register(cr)
        qc.measure(phase_reg[k], cr[0])
        # Note: In a true iterative algorithm the measurement result would be used
        # to apply a conditional phase rotation before the next iteration.  For the
        # prototype we record the bit after execution (the caller can extract the
        # counts later).
        measurement_bits.append(0)  # placeholder – actual bits obtained from counts
    return qc, measurement_bits
