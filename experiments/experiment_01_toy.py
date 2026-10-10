"""
Experiment 01: Golden Test Benchmark for QPE and HHL-Style Quantum Linear Solver.
Evaluates:
- Golden Test A: Diagonal matrix A = diag(1, 2), b = [1, 1]^T
- Golden Test B: 2x2 Hermitian matrix A = [[1, 0.2], [0.2, 2]], b = [1, 1]^T
Saves results table to results/tables/ and figure to results/figures/.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eigenwall.quantum.hhl_solver import quantum_linear_solver
from eigenwall.metrics import state_fidelity, residual_error
from eigenwall.matrix_utils import condition_number

def run_toy_experiments():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 65)
    print("EXPERIMENT 01: TOY LINEAR SYSTEM BENCHMARK")
    print("=" * 65)

    # 1. Golden Test A
    A1 = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex)
    b1 = np.array([1.0, 1.0], dtype=complex)
    x1_analytic = np.array([1.0, 0.5], dtype=complex)
    x1_norm = x1_analytic / np.linalg.norm(x1_analytic)

    print("\n--- Running Golden Test A: diag(1, 2) ---")
    res1 = quantum_linear_solver(
        matrix=A1,
        vector_b=b1,
        phase_qubits=4,
        evolution_time=np.pi / 2.0,
        scaling_constant=0.8,
        shots=8192
    )

    fid1 = state_fidelity(x1_norm, res1["solution_state"])
    res_err1 = residual_error(A1, b1, res1["solution_state"])

    print(f"Analytic state   : {x1_norm}")
    print(f"Quantum state    : {res1['solution_state']}")
    print(f"State Fidelity F : {fid1:.5f}")
    print(f"Success Prob P(1): {res1['success_probability']:.4f}")
    print(f"Residual Error   : {res_err1:.5f}")

    # 2. Golden Test B
    A2 = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=complex)
    b2 = np.array([1.0, 1.0], dtype=complex)
    x2_analytic = np.linalg.solve(A2, b2)
    x2_norm = x2_analytic / np.linalg.norm(x2_analytic)

    print("\n--- Running Golden Test B: [[1, 0.2], [0.2, 2]] ---")
    res2 = quantum_linear_solver(
        matrix=A2,
        vector_b=b2,
        phase_qubits=4,
        shots=8192
    )

    fid2 = state_fidelity(x2_norm, res2["solution_state"])
    res_err2 = residual_error(A2, b2, res2["solution_state"])

    print(f"Classical state  : {x2_norm}")
    print(f"Quantum state    : {res2['solution_state']}")
    print(f"State Fidelity F : {fid2:.5f}")
    print(f"Success Prob P(1): {res2['success_probability']:.4f}")
    print(f"Residual Error   : {res_err2:.5f}")

    # Save summary table
    df = pd.DataFrame([
        {
            "Test": "Golden Test A (diag(1, 2))",
            "Condition_Number": condition_number(A1),
            "Fidelity": fid1,
            "Success_Probability": res1["success_probability"],
            "Residual_Error": res_err1
        },
        {
            "Test": "Golden Test B (Hermitian 2x2)",
            "Condition_Number": condition_number(A2),
            "Fidelity": fid2,
            "Success_Probability": res2["success_probability"],
            "Residual_Error": res_err2
        }
    ])
    csv_path = "results/tables/experiment_01_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved Table] {csv_path}")

    # Plot comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Bar chart for Test A amplitudes
    x_idx = np.arange(2)
    axes[0].bar(x_idx - 0.15, np.abs(x1_norm), width=0.3, label="Analytical", color="#2563eb")
    axes[0].bar(x_idx + 0.15, np.abs(res1["solution_state"]), width=0.3, label="Quantum HHL", color="#10b981")
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(["|0>", "|1>"])
    axes[0].set_title(f"Golden Test A (F = {fid1:.4f})")
    axes[0].set_ylabel("State Amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Bar chart for Test B amplitudes
    axes[1].bar(x_idx - 0.15, np.abs(x2_norm), width=0.3, label="Classical", color="#2563eb")
    axes[1].bar(x_idx + 0.15, np.abs(res2["solution_state"]), width=0.3, label="Quantum HHL", color="#10b981")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(["|0>", "|1>"])
    axes[1].set_title(f"Golden Test B (F = {fid2:.4f})")
    axes[1].set_ylabel("State Amplitude")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    fig_path = "results/figures/figure_01_toy_inversion.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] {fig_path}")

if __name__ == "__main__":
    run_toy_experiments()
