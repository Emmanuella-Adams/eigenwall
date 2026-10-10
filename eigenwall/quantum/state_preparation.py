"""
State preparation routines for encoding vectors into quantum registers.
"""

from typing import Union, List
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def prepare_state(
    qc: QuantumCircuit,
    register: QuantumRegister,
    vector: np.ndarray
) -> np.ndarray:
    """
    Prepare an arbitrary state |b> = sum_i b_i |i> on the given quantum register.

    Args:
        qc: QuantumCircuit instance to append initialization instructions to.
        register: QuantumRegister of size n = ceil(log2(N)).
        vector: Input complex vector of length N <= 2^n.

    Returns:
        norm_padded: The normalized complex statevector of length 2^n.
    """
    n_qubits = len(register)
    dim = 2 ** n_qubits
    vec = np.asarray(vector, dtype=complex)

    if len(vec) > dim:
        raise ValueError(f"Vector dimension {len(vec)} exceeds register capacity {dim}")

    # Pad with zeros if necessary
    padded = np.zeros(dim, dtype=complex)
    padded[:len(vec)] = vec

    norm = np.linalg.norm(padded)
    if norm < 1e-15:
        raise ValueError("Cannot prepare state from zero vector")
    norm_padded = padded / norm

    qc.initialize(norm_padded.tolist(), register)
    return norm_padded
