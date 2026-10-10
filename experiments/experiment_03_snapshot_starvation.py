"""
Experiment 03: Snapshot‑Starvation Regime

This script evaluates the impact of limiting the number of measurement shots on the
quantum HHL solver. By sweeping the `shots` parameter we observe how solution
fidelity, ancilla success probability, and estimated execution time degrade as
the measurement budget becomes scarce – a realistic constraint on near‑term
quantum hardware.

The results are saved as CSV and a Matplotlib figure under the `results/`
directory for inclusion in the final presentation.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
# Add parent directory to sys.path to allow importing the eigenwall package when executing from the experiments folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eigenwall.quantum.hhl_solver import quantum_linear_solver
from eigenwall.quantum.ionq_resources import analyze_ionq_resources as estimate_ionq_resources

# Ensure results directories exist
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# Problem definition – use the same 4‑element ULA matrix from earlier experiments
# (R = covariance matrix) and a steering vector a for the desired look‑direction.
# For reproducibility we reuse the helper from first.py (you may replace with
# explicit definitions if needed).

# Simple 4‑element uniform linear array (ULA) simulation
N = 4
# Inter‑element spacing (wavelengths)
d = 0.5
# Incident angles (degrees) – target and two interferers
theta_target = 18.0
interferers = [-30.0, 50.0]

def steering_vector(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return np.exp(1j * 2 * np.pi * d * np.arange(N) * np.sin(theta))

# Desired steering vector
a = steering_vector(theta_target)

# Construct covariance matrix R = sum_i sigma_i * a_i a_i^H + sigma_n I
sigma_signal = 1.0
sigma_interf = 0.2
sigma_noise = 0.1
R = sigma_signal * np.outer(a, a.conj())
for theta_i in interferers:
    a_i = steering_vector(theta_i)
    R += sigma_interf * np.outer(a_i, a_i.conj())
R += sigma_noise * np.eye(N)

# Right‑hand side vector (desired response) – use a steering vector toward target
b = a.real  # we solve for real‑valued beamformer weights

# Shot values to sweep – range from highly starved to fully sampled
shot_grid = [256, 512, 1024, 2048, 4096, 8192]

# Containers for metrics
fid_list = []
success_prob_list = []
exec_time_ms = []

for shots in shot_grid:
    result = quantum_linear_solver(
        matrix=R,
        vector_b=b,
        phase_qubits=4,
        shots=shots,
        use_statevector=False,  # force shot‑based execution path
    )
    fid_list.append(result["fidelity"])
    success_prob_list.append(result["success_probability"])
    # Estimate IonQ resource‑aware execution time (ms) using the metadata
    resources = estimate_ionq_resources(result["metadata"])  # returns dict with 'estimated_time_ms'
    exec_time_ms.append(resources.get("estimated_time_ms", np.nan))

# Save CSV table
csv_path = "results/tables/snapshot_starvation_metrics.csv"
with open(csv_path, "w", newline="") as f:
    f.write("shots,fidelity,success_probability,estimated_time_ms\n")
    for s, f, p, t in zip(shot_grid, fid_list, success_prob_list, exec_time_ms):
        f.write(f"{s},{f:.6f},{p:.6f},{t:.2f}\n")

# Plot fidelity & success probability vs shots (log‑scale x‑axis)
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.plot(shot_grid, fid_list, marker="o", color="#4A90E2")
plt.xscale("log")
plt.title("Solution Fidelity vs Shots")
plt.xlabel("Measurement Shots (log scale)")
plt.ylabel("Fidelity")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(shot_grid, success_prob_list, marker="s", color="#D0021B")
plt.xscale("log")
plt.title("Ancilla Success Prob. vs Shots")
plt.xlabel("Measurement Shots (log scale)")
plt.ylabel("Success Probability")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
fig_path = "results/figures/snapshot_starvation_curve.png"
plt.savefig(fig_path, dpi=150)
plt.close()

print("Snapshot‑starvation experiment completed. CSV saved to", csv_path)
print("Figure saved to", fig_path)
