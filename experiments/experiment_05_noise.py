"""
Experiment 05: Quantum Noise and Beamforming Robustness.
Investigates how quantum gate noise (depolarizing error) affects:
- Solution fidelity F_noisy
- Interference null depth ND_noisy
- Robustness threshold where adaptive beamforming degrades
Saves:
- Figure 08: Noise vs Beamforming Performance (results/figures/figure_08_noise_performance.png)
- Table: results/tables/experiment_05_noise.csv
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from eigenwall.signal_model import ArrayConfig, steering_vector, generate_received_signal, generate_sinusoid
from eigenwall.covariance import sample_covariance, regularize_covariance
from eigenwall.beamforming import mvdr_weights, quantum_weights_to_beamforming
from eigenwall.quantum.hhl_solver import build_hhl_circuit
from eigenwall.quantum.measurement import sample_and_postselect
from eigenwall.metrics import state_fidelity, null_depth_db

def run_noise_experiment():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 65)
    print("EXPERIMENT 05: QUANTUM NOISE AND ROBUSTNESS")
    print("=" * 65)

    # 2-element array for clean noise scaling
    n_elements = 2
    target_angle = 15.0
    interf_angle = -25.0
    config = ArrayConfig(n_elements=n_elements, wavelength=1.0)

    n_snapshots = 500
    s_target = generate_sinusoid(5.0, n_snapshots, amplitude=1.0)
    s_interf = generate_sinusoid(15.0, n_snapshots, amplitude=2.0)

    X = generate_received_signal(
        array_config=config,
        target_angle=target_angle,
        target_signal=s_target,
        interference_sources=[(interf_angle, s_interf)],
        noise_power=0.5,
        random_seed=42
    )

    R_raw = regularize_covariance(sample_covariance(X), epsilon=0.5)
    lam_max_raw = np.linalg.eigvalsh(R_raw)[-1]
    R = R_raw * (2.0 / lam_max_raw)

    a_target = steering_vector(n_elements, target_angle)
    w_classical = mvdr_weights(R, a_target)
    w_classical_norm = w_classical / np.linalg.norm(w_classical)
    nd_classical = null_depth_db(w_classical, target_angle, interf_angle)

    print(f"Classical Null Depth Ground Truth: {nd_classical:.2f} dB")

    # Build HHL circuit with measurements
    qc_meas, meta = build_hhl_circuit(
        matrix=R,
        vector_b=a_target,
        phase_qubits=3,
        measure_all=True
    )

    error_rates = [0.0, 0.002, 0.005, 0.01, 0.02, 0.04]
    shots = 8192
    records = []

    for p in error_rates:
        print(f"\nSimulating depolarizing error p = {p:.3f} ...")
        if p > 0:
            noise_model = NoiseModel()
            depol_1q = depolarizing_error(p, 1)
            depol_2q = depolarizing_error(p * 2, 2)
            noise_model.add_all_qubit_quantum_error(depol_1q, ['u1', 'u2', 'u3', 'h', 'rx', 'ry', 'rz'])
            noise_model.add_all_qubit_quantum_error(depol_2q, ['cx', 'cz'])
            sim = AerSimulator(noise_model=noise_model)
        else:
            sim = AerSimulator()

        transpiled = transpile(qc_meas, sim)
        job = sim.run(transpiled, shots=shots)
        counts = job.result().get_counts()

        post_res = sample_and_postselect(counts, n_sys=meta["n_sys"], total_shots=shots)
        emp_probs = post_res["empirical_probabilities"][:n_elements]

        # Reconstruct empirical amplitude vector
        # (magnitudes from sqrt of probabilities, phases aligned with classical target)
        q_state_noisy = np.sqrt(np.maximum(0.0, emp_probs))
        norm_q = np.linalg.norm(q_state_noisy)
        if norm_q > 1e-12:
            q_state_noisy /= norm_q
        else:
            q_state_noisy = np.ones(n_elements) / np.sqrt(n_elements)

        # Apply target phase alignment
        q_state_noisy = q_state_noisy * np.exp(1j * np.angle(w_classical_norm))

        fid = state_fidelity(w_classical_norm, q_state_noisy)
        w_phys_noisy = quantum_weights_to_beamforming(q_state_noisy, a_target)
        nd_noisy = null_depth_db(w_phys_noisy, target_angle, interf_angle)

        print(f"  p = {p:.3f} -> Fidelity F = {fid:.4f}, Null Depth = {nd_noisy:.2f} dB")
        records.append({
            "Error_Rate_p": p,
            "Fidelity": fid,
            "Null_Depth_dB": nd_noisy,
            "Success_Probability": post_res["success_probability"]
        })

    df = pd.DataFrame(records)
    csv_path = "results/tables/experiment_05_noise.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved Table] {csv_path}")

    # Figure 08: Noise vs Beamforming Performance
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))

    color_fid = "#059669"
    ax1.set_xlabel("Depolarizing Gate Error Rate p", fontsize=11)
    ax1.set_ylabel("Solution Fidelity F", color=color_fid, fontsize=11)
    ax1.plot(df["Error_Rate_p"], df["Fidelity"], marker="o", linewidth=2.2, color=color_fid, label="Fidelity F")
    ax1.tick_params(axis="y", labelcolor=color_fid)
    ax1.set_ylim([0.4, 1.05])
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_nd = "#ea580c"
    ax2.set_ylabel("Null Depth (dB)", color=color_nd, fontsize=11)
    ax2.plot(df["Error_Rate_p"], df["Null_Depth_dB"], marker="^", linestyle="--", linewidth=2.0, color=color_nd, label="Null Depth (dB)")
    ax2.axhline(nd_classical, color="gray", linestyle=":", label="Classical ND")
    ax2.tick_params(axis="y", labelcolor=color_nd)

    plt.title("EigenWall: Noise Robustness vs Beamforming Null Depth", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_path = "results/figures/figure_08_noise_performance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] {fig_path}")

if __name__ == "__main__":
    run_noise_experiment()
