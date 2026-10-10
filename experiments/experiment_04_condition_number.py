"""
Experiment 04: Condition Number Sweep.
Investigates how the condition number kappa = lambda_max / lambda_min affects:
- Quantum linear system solution fidelity F
- Inversion sensitivity and success probability
Saves:
- Figure 07: Solution Fidelity vs Condition Number (results/figures/figure_07_condition_number.png)
- Table: results/tables/experiment_04_condition_number.csv
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eigenwall.quantum.hhl_solver import quantum_linear_solver
from eigenwall.matrix_utils import condition_number
from eigenwall.metrics import state_fidelity

def run_condition_number_experiment():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 65)
    print("EXPERIMENT 04: CONDITION NUMBER SWEEP")
    print("=" * 65)

    # Vary lambda_min while keeping lambda_max = 2.0
    lam_max = 2.0
    lam_min_values = [1.5, 1.0, 0.6, 0.4, 0.25, 0.15]
    b = np.array([1.0, 1.0], dtype=complex)

    records = []

    for l_min in lam_min_values:
        A = np.diag([l_min, lam_max]).astype(complex)
        kappa = condition_number(A)
        print(f"\nTesting lambda_min = {l_min:.2f} (kappa = {kappa:.2f})...")

        res = quantum_linear_solver(
            matrix=A,
            vector_b=b,
            phase_qubits=4,
            evolution_time=np.pi / lam_max,
            scaling_constant=0.9 * l_min,
            shots=4096
        )

        fid = res["fidelity"]
        p_succ = res["success_probability"]
        print(f"  kappa = {kappa:.2f} -> Fidelity F = {fid:.4f}, P(succ) = {p_succ:.4f}")

        records.append({
            "Lambda_Min": l_min,
            "Lambda_Max": lam_max,
            "Condition_Number": kappa,
            "Solution_Fidelity": fid,
            "Success_Probability": p_succ
        })

    df = pd.DataFrame(records)
    csv_path = "results/tables/experiment_04_condition_number.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved Table] {csv_path}")

    # Figure 07: Solution Fidelity vs Condition Number
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(df["Condition_Number"], df["Solution_Fidelity"], marker="o", linewidth=2.2, color="#7c3aed", label="Fidelity F")
    plt.axhline(1.0, color="gray", linestyle=":", alpha=0.7)

    plt.title("EigenWall: Solution Fidelity vs Condition Number kappa", fontsize=12, fontweight="bold")
    plt.xlabel("Condition Number kappa = lambda_max / lambda_min", fontsize=11)
    plt.ylabel("Solution Fidelity F", fontsize=11)
    plt.ylim([0.7, 1.02])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")
    plt.tight_layout()

    fig_path = "results/figures/figure_07_condition_number.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] {fig_path}")

if __name__ == "__main__":
    run_condition_number_experiment()
