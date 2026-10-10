"""
IonQ Trapped-Ion Hardware Resource Estimation and Coherence Budget Analysis.
Analyzes native gate compilation (R_XX, R_Z, R_X, R_Y) and checks physical coherence limits.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def analyze_ionq_resources(
    circuit: QuantumCircuit,
    t_2q_us: float = 200.0,
    t_1q_us: float = 10.0,
    t2_coherence_us: float = 1_000_000.0  # 1.0 second T2 for 171Yb+
) -> Dict[str, Any]:
    """
    Transpile circuit to IonQ native basis gates and compute physical resource budget:
    - 2-qubit native entangling gate: R_XX (Molmer-Sorensen)
    - 1-qubit native rotation gates: RX, RY, RZ, H
    - Estimated circuit duration vs T2 coherence time
    """
    ionq_basis = ['rxx', 'rx', 'ry', 'rz', 'h']
    sim = AerSimulator()

    try:
        transpiled = transpile(circuit, sim, basis_gates=ionq_basis, optimization_level=2)
    except Exception:
        transpiled = transpile(circuit, sim, optimization_level=1)

    gate_counts = transpiled.count_ops()
    n_rxx = gate_counts.get('rxx', 0)
    # Total 1-qubit gates
    n_1q = sum(v for k, v in gate_counts.items() if k != 'rxx')
    total_gates = sum(gate_counts.values())

    # Duration estimates
    duration_2q_us = n_rxx * t_2q_us
    duration_1q_us = n_1q * t_1q_us
    total_duration_us = duration_2q_us + duration_1q_us
    total_duration_ms = total_duration_us / 1000.0

    coherence_margin = t2_coherence_us / total_duration_us if total_duration_us > 0 else np.inf
    estimated_coherence_decay = float(np.exp(-total_duration_us / t2_coherence_us))

    res = {
        "num_qubits": transpiled.num_qubits,
        "circuit_depth": transpiled.depth(),
        "total_gates": total_gates,
        "rxx_2q_gates": n_rxx,
        "single_qubit_gates": n_1q,
        "gate_breakdown": dict(gate_counts),
        "est_duration_ms": total_duration_ms,
        "t2_coherence_ms": t2_coherence_us / 1000.0,
        "coherence_margin_ratio": coherence_margin,
        "coherence_survival_prob": estimated_coherence_decay,
        "all_to_all_connectivity_used": True
    }
    return res


def export_resource_table(resource_dict: Dict[str, Any], output_path: str = None) -> pd.DataFrame:
    """Format and save IonQ resource budget table."""
    data = [
        {"Resource / Metric": "Total Qubits", "Value": str(resource_dict["num_qubits"]), "Unit": "qubits"},
        {"Resource / Metric": "Circuit Depth (IonQ Basis)", "Value": str(resource_dict["circuit_depth"]), "Unit": "layers"},
        {"Resource / Metric": "Total Gate Count", "Value": str(resource_dict["total_gates"]), "Unit": "gates"},
        {"Resource / Metric": "Native 2-Qubit R_XX Gates", "Value": str(resource_dict["rxx_2q_gates"]), "Unit": "MS gates"},
        {"Resource / Metric": "Single-Qubit Rotations", "Value": str(resource_dict["single_qubit_gates"]), "Unit": "gates"},
        {"Resource / Metric": "Estimated Execution Time", "Value": f"{resource_dict['est_duration_ms']:.2f}", "Unit": "ms"},
        {"Resource / Metric": "Trapped-Ion T2 Coherence Time", "Value": f"{resource_dict['t2_coherence_ms']:.1f}", "Unit": "ms"},
        {"Resource / Metric": "Coherence Safety Margin", "Value": f"{resource_dict['coherence_margin_ratio']:.1f}x", "Unit": "ratio"},
        {"Resource / Metric": "Coherence Survival Probability", "Value": f"{resource_dict['coherence_survival_prob'] * 100:.2f}%", "Unit": "prob"},
    ]
    df = pd.DataFrame(data)
    if output_path:
        df.to_csv(output_path, index=False)
    return df
