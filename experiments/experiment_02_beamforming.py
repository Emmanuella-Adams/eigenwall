"""
Experiment 02: End-to-End Quantum-Assisted Adaptive Beamforming Benchmark.
Comprehensive Evaluation:
- Antenna signal simulation (N=4 ULA, Target: 20 deg, Multi-Interferer: -30 deg & +50 deg)
- Sample covariance estimation & spectral scaling
- Classical MVDR vs Quantum HHL-assisted beamforming
- Output SINR (Signal-to-Interference-plus-Noise Ratio) & Multi-Null Depths
- Trapped-Ion (IonQ) Native Gate Resource & Coherence Budget Analysis
- Dual-Panel Deliverable: Cartesian (dB) + Polar Radiation Pattern (Figure 5)
Saves:
- Figure 02: Covariance Matrix Heatmap
- Figure 03: Eigenvalue Spectrum
- Figure 05: Classical vs Quantum Beam Pattern (Cartesian + Polar)
- Tables: metrics_dashboard.csv & ionq_resource_budget.csv
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eigenwall.signal_model import ArrayConfig, steering_vector, generate_received_signal, generate_sinusoid
from eigenwall.covariance import sample_covariance, regularize_covariance, check_covariance_properties
from eigenwall.beamforming import mvdr_weights, beam_pattern, quantum_weights_to_beamforming
from eigenwall.quantum.hhl_solver import quantum_linear_solver, build_hhl_circuit
from eigenwall.quantum.ionq_resources import analyze_ionq_resources, export_resource_table
from eigenwall.metrics import (
    state_fidelity, residual_error, null_depth_db, beam_direction_error,
    array_response_power, compute_sinr_db, multi_null_depth_db
)


def run_beamforming_experiment():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 75)
    print("EXPERIMENT 02: END-TO-END QUANTUM ADAPTIVE BEAMFORMING BENCHMARK")
    print("=" * 75)

    # 1. Configuration
    n_elements = 4
    target_angle = 20.0
    interf_angles = [-30.0, 50.0]
    wavelength = 1.0
    config = ArrayConfig(n_elements=n_elements, wavelength=wavelength)

    n_snapshots = 1000
    target_sig = generate_sinusoid(frequency=5.0, n_snapshots=n_snapshots, amplitude=1.0)
    interf_sig1 = generate_sinusoid(frequency=18.0, n_snapshots=n_snapshots, amplitude=1.8)
    interf_sig2 = generate_sinusoid(frequency=32.0, n_snapshots=n_snapshots, amplitude=1.5)

    print(f"\n[Scenario]")
    print(f"  Array Architecture:    {n_elements}-element Uniform Linear Array (d = lambda/2)")
    print(f"  Target Spacecraft:     {target_angle} deg (power = 1.0)")
    print(f"  Interferer 1 (RFI):    {interf_angles[0]} deg (power = 3.24, INR ~ +6 dB)")
    print(f"  Interferer 2 (Sat):    {interf_angles[1]} deg (power = 2.25, INR ~ +4.5 dB)")
    print(f"  Snapshots Synthesized: {n_snapshots}")

    # 2. Received Signal Simulation & Covariance
    X = generate_received_signal(
        array_config=config,
        target_angle=target_angle,
        target_signal=target_sig,
        interference_sources=[(interf_angles[0], interf_sig1), (interf_angles[1], interf_sig2)],
        noise_power=0.8,
        random_seed=42
    )

    # Isolated Interference + Noise for SINR calculation
    X_in = generate_received_signal(
        array_config=config,
        target_angle=target_angle,
        target_signal=np.zeros(n_snapshots, dtype=complex),
        interference_sources=[(interf_angles[0], interf_sig1), (interf_angles[1], interf_sig2)],
        noise_power=0.8,
        random_seed=42
    )
    R_in = sample_covariance(X_in)

    R_raw = sample_covariance(X)
    # Diagonal loading (robust regularization)
    R_loaded = regularize_covariance(R_raw, epsilon=1.0)

    # Scale covariance matrix for QPE dynamic range matching:
    # Scale invariance: (c * R)^-1 * a propto R^-1 * a (weights are mathematically identical)
    lam_max_raw = float(np.linalg.eigvalsh(R_loaded)[-1])
    scale_factor = 2.0 / lam_max_raw
    R = R_loaded * scale_factor

    props = check_covariance_properties(R)

    print(f"\n[Covariance Analysis (Scaled for QPE Spectrum)]")
    print(f"  Hermitian:       {props['is_hermitian']} (err: {props['hermiticity_error']:.2e})")
    print(f"  Positive Def:    {props['is_positive_definite']}")
    print(f"  Eigenvalues:     {[round(float(e), 4) for e in props['eigenvalues']]}")
    print(f"  Condition kappa: {props['condition_number']:.2f}")

    # 3. Target Steering Vector
    a_target = steering_vector(n_elements, target_angle, wavelength=wavelength)

    # 4. Classical MVDR Solver
    w_classical = mvdr_weights(R, a_target)
    w_classical_norm = w_classical / np.linalg.norm(w_classical)

    print(f"\n[Classical MVDR]")
    print(f"  Weights: {w_classical}")
    print(f"  Constraint w^H * a: {np.vdot(w_classical, a_target):.4f} (Required: 1.0)")

    # 5. Quantum Linear Solver (HHL)
    print(f"\n[Quantum HHL Solver Execution]")
    q_res = quantum_linear_solver(
        matrix=R,
        vector_b=a_target,
        phase_qubits=4,
        shots=8192
    )

    w_quantum_state = q_res["solution_state"]
    fidelity = state_fidelity(w_classical_norm, w_quantum_state)
    q_res_err = residual_error(R, a_target, w_quantum_state)
    w_quantum_phys = quantum_weights_to_beamforming(w_quantum_state, a_target)

    print(f"  State Fidelity F:  {fidelity:.5f}")
    print(f"  Success Prob P(1): {q_res['success_probability']:.4f}")
    print(f"  Residual Error:    {q_res_err:.5f}")
    print(f"  Reconstructed w_q: {w_quantum_phys}")

    # 6. Beam Pattern Computation
    angles_grid = np.linspace(-90.0, 90.0, 721)
    _, p_c, db_c = beam_pattern(w_classical, angles_grid, wavelength=wavelength)
    _, p_q, db_q = beam_pattern(w_quantum_phys, angles_grid, wavelength=wavelength)

    # 7. Metrics & SINR Evaluation
    sinr_c = compute_sinr_db(w_classical, a_target, R_in, target_power=1.0)
    sinr_q = compute_sinr_db(w_quantum_phys, a_target, R_in, target_power=1.0)

    null_depths_c = multi_null_depth_db(w_classical, target_angle, interf_angles, wavelength=wavelength)
    null_depths_q = multi_null_depth_db(w_quantum_phys, target_angle, interf_angles, wavelength=wavelength)

    g_target_c = array_response_power(w_classical, target_angle, wavelength=wavelength)
    g_target_q = array_response_power(w_quantum_phys, target_angle, wavelength=wavelength)

    beam_err = beam_direction_error(w_classical, w_quantum_phys, angles_grid, wavelength=wavelength)
    peak_angle_c = angles_grid[np.argmax(db_c)]
    peak_angle_q = angles_grid[np.argmax(db_q)]

    # 8. Trapped-Ion (IonQ) Hardware Resource Budget
    ionq_budget = analyze_ionq_resources(q_res["circuit"])
    df_ionq = export_resource_table(ionq_budget, "results/tables/ionq_resource_budget.csv")
    print(f"\n[IonQ Resource Budget Table Saved: results/tables/ionq_resource_budget.csv]")
    print(f"  Native R_XX 2Q Gates:      {ionq_budget['rxx_2q_gates']}")
    print(f"  Estimated Circuit Time:    {ionq_budget['est_duration_ms']:.2f} ms")
    print(f"  Trapped-Ion Coherence T2:  {ionq_budget['t2_coherence_ms']:.1f} ms")
    print(f"  Coherence Safety Margin:   {ionq_budget['coherence_margin_ratio']:.1f}x")

    # Metrics table
    metrics_data = [
        {"Metric": "Solution Fidelity F", "Classical": "1.0000", "Quantum": f"{fidelity:.4f}", "Difference": f"{1.0 - fidelity:.4f}"},
        {"Metric": "Output SINR (dB)", "Classical": f"{sinr_c:.2f} dB", "Quantum": f"{sinr_q:.2f} dB", "Difference": f"{abs(sinr_c - sinr_q):.2f} dB"},
        {"Metric": "Target Power Gain", "Classical": f"{g_target_c:.4f}", "Quantum": f"{g_target_q:.4f}", "Difference": f"{abs(g_target_c - g_target_q):.4f}"},
        {"Metric": "Null Depth Interf 1 (-30°)", "Classical": f"{null_depths_c[interf_angles[0]]:.2f} dB", "Quantum": f"{null_depths_q[interf_angles[0]]:.2f} dB", "Difference": f"{abs(null_depths_c[interf_angles[0]] - null_depths_q[interf_angles[0]]):.2f} dB"},
        {"Metric": "Null Depth Interf 2 (+50°)", "Classical": f"{null_depths_c[interf_angles[1]]:.2f} dB", "Quantum": f"{null_depths_q[interf_angles[1]]:.2f} dB", "Difference": f"{abs(null_depths_c[interf_angles[1]] - null_depths_q[interf_angles[1]]):.2f} dB"},
        {"Metric": "Peak Beam Direction", "Classical": f"{peak_angle_c:.1f} deg", "Quantum": f"{peak_angle_q:.1f} deg", "Difference": f"{beam_err:.2f} deg"},
        {"Metric": "Residual Error", "Classical": "0.0000", "Quantum": f"{q_res_err:.4f}", "Difference": f"{q_res_err:.4f}"},
        {"Metric": "Quantum Qubits", "Classical": "—", "Quantum": str(q_res['circuit'].num_qubits), "Difference": "—"},
        {"Metric": "Native R_XX Gates (IonQ)", "Classical": "—", "Quantum": str(ionq_budget['rxx_2q_gates']), "Difference": "—"},
        {"Metric": "Circuit Depth", "Classical": "—", "Quantum": str(q_res['circuit'].depth()), "Difference": "—"},
    ]
    df_metrics = pd.DataFrame(metrics_data)
    dashboard_path = "results/tables/metrics_dashboard.csv"
    df_metrics.to_csv(dashboard_path, index=False)
    print(f"\n[Saved Metrics Dashboard] {dashboard_path}")
    print("\n" + df_metrics.to_string(index=False))

    # 9. Deliverable Visualizations

    # Figure 2: Covariance Matrix Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im0 = axes[0].imshow(np.abs(R), cmap="viridis")
    axes[0].set_title("Covariance Magnitude |R|", fontsize=11, fontweight="bold")
    axes[0].set_xticks(range(n_elements))
    axes[0].set_yticks(range(n_elements))
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.angle(R), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Covariance Phase arg(R)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(range(n_elements))
    axes[1].set_yticks(range(n_elements))
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig("results/figures/figure_02_covariance.png", dpi=300)
    plt.close()

    # Figure 3: Eigenvalue Spectrum
    plt.figure(figsize=(6, 4))
    plt.stem(range(1, n_elements + 1), props["eigenvalues"], basefmt=" ")
    plt.title(f"Covariance Eigenvalue Spectrum (kappa = {props['condition_number']:.2f})", fontsize=11, fontweight="bold")
    plt.xlabel("Eigenvalue Index i")
    plt.ylabel("Eigenvalue lambda_i")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/figures/figure_03_eigenvalues.png", dpi=300)
    plt.close()

    # Figure 5: DUAL-PANEL DELIVERABLE (Cartesian dB + Polar Radiation Pattern)
    fig = plt.figure(figsize=(14, 5.5))

    # Panel 1: Cartesian dB Plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(angles_grid, db_c, label="Classical MVDR Ground Truth", color="#1e3a8a", linewidth=2.2)
    ax1.plot(angles_grid, db_q, label=f"Quantum HHL Assisted (F = {fidelity:.3f})", color="#dc2626", linestyle="--", linewidth=2.0)
    ax1.axvline(target_angle, color="#16a34a", linestyle=":", linewidth=1.5, label=f"Target ({target_angle}°)")
    ax1.axvline(interf_angles[0], color="#ea580c", linestyle=":", linewidth=1.5, label=f"Interf 1 ({interf_angles[0]}°)")
    ax1.axvline(interf_angles[1], color="#9333ea", linestyle=":", linewidth=1.5, label=f"Interf 2 ({interf_angles[1]}°)")

    ax1.set_title("Cartesian Array Response (dB vs. Angle)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Angle of Arrival (degrees)", fontsize=11)
    ax1.set_ylabel("Normalized Response (dB)", fontsize=11)
    ax1.set_ylim([-50.0, 5.0])
    ax1.set_xlim([-90.0, 90.0])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Panel 2: Polar Radiation Pattern (The RF Gold Standard)
    ax2 = fig.add_subplot(1, 2, 2, polar=True)
    # Convert angles to radians (0 deg broadside is polar pi/2)
    angles_rad = np.deg2rad(angles_grid)
    # Clip dB pattern to range [-45, 0] for clean polar visualization
    db_c_polar = np.maximum(-45.0, db_c) + 45.0
    db_q_polar = np.maximum(-45.0, db_q) + 45.0

    ax2.plot(angles_rad + np.pi/2, db_c_polar, label="Classical MVDR", color="#1e3a8a", linewidth=2.0)
    ax2.plot(angles_rad + np.pi/2, db_q_polar, label="Quantum HHL", color="#dc2626", linestyle="--", linewidth=1.8)

    # Plot markers for target and interferers
    target_rad = np.deg2rad(target_angle) + np.pi/2
    interf1_rad = np.deg2rad(interf_angles[0]) + np.pi/2
    interf2_rad = np.deg2rad(interf_angles[1]) + np.pi/2

    ax2.plot([target_rad, target_rad], [0, 45.0], color="#16a34a", linestyle=":", linewidth=1.5)
    ax2.plot([interf1_rad, interf1_rad], [0, 45.0], color="#ea580c", linestyle=":", linewidth=1.5)
    ax2.plot([interf2_rad, interf2_rad], [0, 45.0], color="#9333ea", linestyle=":", linewidth=1.5)

    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_thetalim(0, np.pi)
    ax2.set_title("Polar Radiation Pattern (0 dB at Outer Ring)", fontsize=12, fontweight="bold", pad=15)
    ax2.legend(loc="lower left", fontsize=9)

    fig.suptitle(f"EigenWall: Classical vs. Quantum Adaptive Beamforming (N={n_elements}, Multi-Interferer)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    plt.savefig("results/figures/figure_05_classical_vs_quantum.png", dpi=300)
    plt.savefig("results/figures/classical_vs_quantum.png", dpi=300)
    plt.close()

    print(f"\n[Generated Dual-Panel Figures]")
    print(f"  results/figures/figure_05_classical_vs_quantum.png (Cartesian + Polar)")
    print(f"  results/figures/classical_vs_quantum.png")


if __name__ == "__main__":
    run_beamforming_experiment()
