"""
Quantum algorithms and subroutines for EigenWall:
State preparation, QPE, HHL-style linear solver, and readout/measurement.
"""

from eigenwall.quantum.state_preparation import prepare_state
from eigenwall.quantum.qpe import build_qpe_circuit, decode_phase, run_qpe
from eigenwall.quantum.hhl_solver import quantum_linear_solver, build_hhl_circuit
from eigenwall.quantum.measurement import sample_and_postselect

__all__ = [
    "prepare_state",
    "build_qpe_circuit",
    "decode_phase",
    "run_qpe",
    "quantum_linear_solver",
    "build_hhl_circuit",
    "sample_and_postselect",
]
