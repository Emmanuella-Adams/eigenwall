"""
State preparation utilities for EigenWall.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def block_encode(
    qc: QuantumCircuit,
    register: QuantumRegister,
    vector: np.ndarray,
) -> np.ndarray:
    """Encode a classical vector into a quantum register using binary‑tree amplitude encoding.
    Works for modest sizes (N <= 12)."""
    n_qubits = len(register)
    dim = 2 ** n_qubits
    vec = np.asarray(vector, dtype=complex)
    if len(vec) > dim:
        raise ValueError(f"Vector dimension {len(vec)} exceeds register capacity {dim}")
    padded = np.zeros(dim, dtype=complex)
    padded[: len(vec)] = vec
    norm = np.linalg.norm(padded)
    if norm < 1e-15:
        raise ValueError("Cannot encode zero vector")
    state = padded / norm

    def apply_rotation(start: int, size: int, qubit_idx: int):
        """Recursively encode amplitudes for the sub‑vector ``state[start:start+size]``."""
        if size == 1:
            return
        half = size // 2
        norm0 = np.linalg.norm(state[start : start + half])
        norm1 = np.linalg.norm(state[start + half : start + size])
        theta = 2 * np.arccos(norm0) if norm0 + norm1 > 0 else 0.0
        qc.ry(theta, register[qubit_idx])
        if norm0 > 1e-12:
            with qc.if_test((register[qubit_idx], 0)):
                apply_rotation(start, half, qubit_idx + 1)
        if norm1 > 1e-12:
            with qc.if_test((register[qubit_idx], 1)):
                apply_rotation(start + half, half, qubit_idx + 1)

    apply_rotation(0, dim, 0)
    return state


def prepare_state(
    qc: QuantumCircuit,
    register: QuantumRegister,
    vector: np.ndarray,
) -> np.ndarray:
    """Legacy wrapper forwarding to block_encode."""
    return block_encode(qc, register, vector)
