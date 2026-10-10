"""
Experiment 03: QPE Precision Sweep.
Investigates how the number of phase qubits m in {2, 3, 4, 5} affects:
- Eigenvalue estimation error epsilon_lambda
- Linear system solution fidelity F
Saves:
- Figure 06: Fidelity and Eigenvalue Error vs QPE Precision (results/figures/figure_06_qpe_precision.png)
- Table: results/tables/experiment_03_qpe_precision.csv
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eigenwall.quantum.qpe import run_qpe
from eigenwall.quantum.hhl_solver import quantum_linear_solver
from eigenwall.metrics import state_fidelity

def run_qpe_precision_experiment():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 65)
    print("EXPERIMENT 03: QPE PRECISION SWEEP (m in {2, 3, 4, 5})")
    print("=" * 65)

    # Use 2x2 test matrix with known spectrum
    A = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=complex)
    b = np.array([1.0, 1.0], dtype=complex)
    eigvals, eigvecs = np.linalg.eigh(A)
    lam_0, lam_1 = eigvals[0], eigvals[1]
    t_val = np.pi / lam_1

    m_values = [2, 3, 4, 5]
    records = []

    for m in m_values:
        print(f"\nEvaluating m = {m} phase qubits...")

        # 1. QPE evaluation on eigenstate 0
        _, qpe_res = run_qpe(
            A, t_val, m,
            input_state_vector=eigvecs[:, 0],
            lambda_true=lam_0,
            shots=4096
        )
        lam_err = qpe_res["error"]

        # 2. Linear solver fidelity
        solver_res = quantum_linear_solver(
            matrix=A,
            vector_b=b,
            phase_qubits=m,
            evolution_time=t_val,
            shots=4096
        )
        fid = solver_res["fidelity"]
        p_succ = solver_res["success_probability"]

        print(f"  m = {m}: Eigenvalue Error = {lam_err:.4f}, Fidelity F = {fid:.4f}, P(succ) = {p_succ:.4f}")
        records.append({
            "Phase_Qubits": m,
            "Bin_Spacing_Lambda": qpe_res["bin_spacing_lambda"],
            "Eigenvalue_Error": lam_err,
            "Solution_Fidelity": fid,
            "Success_Probability": p_succ,
            "Circuit_Depth": solver_res["circuit"].depth()
        })

    df = pd.DataFrame(records)
    csv_path = "results/tables/experiment_03_qpe_precision.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved Table] {csv_path}")

    # Figure 06: Fidelity & Eigenvalue Error vs QPE Precision
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))

    color_fid = "#2563eb"
    ax1.set_xlabel("Phase Evaluation Qubits (m)", fontsize=11)
    ax1.set_ylabel("Solution Fidelity F", color=color_fid, fontsize=11)
    ax1.plot(df["Phase_Qubits"], df["Solution_Fidelity"], marker="o", linewidth=2.2, color=color_fid, label="Fidelity F")
    ax1.tick_params(axis="y", labelcolor=color_fid)
    ax1.set_ylim([0.8, 1.02])
    ax1.set_xticks(m_values)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_err = "#dc2626"
    ax2.set_ylabel("Eigenvalue Error |lambda_est - lambda_true|", color=color_err, fontsize=11)
    ax2.plot(df["Phase_Qubits"], df["Eigenvalue_Error"], marker="s", linestyle="--", linewidth=2.0, color=color_err, label="Eigenvalue Error")
    ax2.tick_params(axis="y", labelcolor=color_err)

    plt.title("EigenWall: Solution Quality vs QPE Register Precision", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_path = "results/figures/figure_06_qpe_precision.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] {fig_path}")

if __name__ == "__main__":
    run_qpe_precision_experiment()
