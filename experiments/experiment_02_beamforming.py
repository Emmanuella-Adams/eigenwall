"""
Experiment 02: End-to-End Quantum-Assisted Adaptive Beamforming.
Evaluates:
- Antenna signal simulation (N=4 ULA, target=20 deg, interference=-30 deg)
- Sample covariance estimation and regularization
- Classical MVDR vs Quantum HHL-assisted beamforming
- Beam pattern comparison and metrics dashboard
Saves:
- Figure 02: Covariance matrix
- Figure 03: Eigenvalue spectrum
- Figure 05: Classical vs Quantum beam pattern (Figure 5)
- Metrics dashboard CSV table
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
from eigenwall.quantum.hhl_solver import quantum_linear_solver
from eigenwall.metrics import state_fidelity, residual_error, null_depth_db, beam_direction_error, array_response_power


def run_beamforming_experiment():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 02: END-TO-END QUANTUM ADAPTIVE BEAMFORMING")
    print("=" * 70)

    # 1. Configuration
    n_elements = 4
    target_angle = 20.0
    interf_angle = -30.0
    wavelength = 1.0
    config = ArrayConfig(n_elements=n_elements, wavelength=wavelength)

    n_snapshots = 1000
    target_sig = generate_sinusoid(frequency=5.0, n_snapshots=n_snapshots, amplitude=1.0)
    interf_sig = generate_sinusoid(frequency=18.0, n_snapshots=n_snapshots, amplitude=1.8)

    print(f"\n[Scenario]")
    print(f"  Array: {n_elements}-element Uniform Linear Array (d = lambda/2)")
    print(f"  Target Spacecraft: {target_angle} deg (power = 1.0)")
    print(f"  Interferer:        {interf_angle} deg (power = 3.24, INR ~ +6 dB)")
    print(f"  Snapshots:         {n_snapshots}")

    # 2. Received Signal Simulation & Covariance
    X = generate_received_signal(
        array_config=config,
        target_angle=target_angle,
        target_signal=target_sig,
        interference_sources=[(interf_angle, interf_sig)],
        noise_power=0.8,
        random_seed=42
    )

    R_raw = sample_covariance(X)
    # Diagonal loading (robust adaptive beamforming regularization)
    R_loaded = regularize_covariance(R_raw, epsilon=1.0)

    # Scale covariance matrix for QPE dynamic range matching:
    # Scale invariance: (c * R)^-1 * a propto R^-1 * a (weights are mathematically identical)
    lam_max_raw = np.linalg.eigvalsh(R_loaded)[-1]
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
    print(f"  w^H * a: {np.vdot(w_classical, a_target):.4f} (Constraint: 1.0)")

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

    # 7. Metrics Dashboard Calculation
    nd_c = null_depth_db(w_classical, target_angle, interf_angle, wavelength=wavelength)
    nd_q = null_depth_db(w_quantum_phys, target_angle, interf_angle, wavelength=wavelength)

    g_target_c = array_response_power(w_classical, target_angle, wavelength=wavelength)
    g_target_q = array_response_power(w_quantum_phys, target_angle, wavelength=wavelength)

    g_interf_c = array_response_power(w_classical, interf_angle, wavelength=wavelength)
    g_interf_q = array_response_power(w_quantum_phys, interf_angle, wavelength=wavelength)

    beam_err = beam_direction_error(w_classical, w_quantum_phys, angles_grid, wavelength=wavelength)

    peak_angle_c = angles_grid[np.argmax(db_c)]
    peak_angle_q = angles_grid[np.argmax(db_q)]

    # Metrics table
    metrics_data = [
        {"Metric": "Solution Fidelity F", "Classical": "1.0000", "Quantum": f"{fidelity:.4f}", "Difference": f"{1.0 - fidelity:.4f}"},
        {"Metric": "Target Power Gain", "Classical": f"{g_target_c:.4f}", "Quantum": f"{g_target_q:.4f}", "Difference": f"{abs(g_target_c - g_target_q):.4f}"},
        {"Metric": "Interference Power Gain", "Classical": f"{g_interf_c:.2e}", "Quantum": f"{g_interf_q:.2e}", "Difference": f"{abs(g_interf_c - g_interf_q):.2e}"},
        {"Metric": "Null Depth (dB)", "Classical": f"{nd_c:.2f} dB", "Quantum": f"{nd_q:.2f} dB", "Difference": f"{abs(nd_c - nd_q):.2f} dB"},
        {"Metric": "Peak Beam Direction", "Classical": f"{peak_angle_c:.1f} deg", "Quantum": f"{peak_angle_q:.1f} deg", "Difference": f"{beam_err:.2f} deg"},
        {"Metric": "Residual Error", "Classical": "0.0000", "Quantum": f"{q_res_err:.4f}", "Difference": f"{q_res_err:.4f}"},
        {"Metric": "Quantum Qubits", "Classical": "—", "Quantum": str(q_res['circuit'].num_qubits), "Difference": "—"},
        {"Metric": "Circuit Depth", "Classical": "—", "Quantum": str(q_res['circuit'].depth()), "Difference": "—"},
    ]
    df_metrics = pd.DataFrame(metrics_data)
    dashboard_path = "results/tables/metrics_dashboard.csv"
    df_metrics.to_csv(dashboard_path, index=False)
    print("\n" + df_metrics.to_string(index=False))

    # 8. Visualization Generation

    # Figure 2: Covariance Matrix Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im0 = axes[0].imshow(np.abs(R), cmap="viridis")
    axes[0].set_title("Covariance Magnitude |R|")
    axes[0].set_xticks(range(n_elements))
    axes[0].set_yticks(range(n_elements))
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.angle(R), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Covariance Phase arg(R)")
    axes[1].set_xticks(range(n_elements))
    axes[1].set_yticks(range(n_elements))
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig("results/figures/figure_02_covariance.png", dpi=300)
    plt.close()

    # Figure 3: Eigenvalue Spectrum
    plt.figure(figsize=(6, 4))
    plt.stem(range(1, n_elements + 1), props["eigenvalues"], basefmt=" ")
    plt.title(f"Covariance Eigenvalue Spectrum (kappa = {props['condition_number']:.2f})")
    plt.xlabel("Eigenvalue Index i")
    plt.ylabel("Eigenvalue lambda_i")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/figures/figure_03_eigenvalues.png", dpi=300)
    plt.close()

    # Figure 5: Classical vs Quantum Beam Pattern (Primary Deliverable)
    plt.figure(figsize=(9, 5.2))
    plt.plot(angles_grid, db_c, label="Classical MVDR Ground Truth", color="#1e3a8a", linewidth=2.2)
    plt.plot(angles_grid, db_q, label=f"Quantum HHL Assisted (F = {fidelity:.3f})", color="#dc2626", linestyle="--", linewidth=2.0)
    plt.axvline(target_angle, color="#16a34a", linestyle=":", linewidth=1.5, label=f"Target ({target_angle}°)")
    plt.axvline(interf_angle, color="#f97316", linestyle=":", linewidth=1.5, label=f"Interference ({interf_angle}°)")

    plt.title(f"EigenWall: Classical vs Quantum MVDR Radiation Pattern (N={n_elements})", fontsize=12, fontweight="bold")
    plt.xlabel("Angle of Arrival (degrees)", fontsize=11)
    plt.ylabel("Normalized Array Response (dB)", fontsize=11)
    plt.ylim([-50.0, 5.0])
    plt.xlim([-90.0, 90.0])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    # Save to both standard names
    plt.savefig("results/figures/figure_05_classical_vs_quantum.png", dpi=300)
    plt.savefig("results/figures/classical_vs_quantum.png", dpi=300)
    plt.close()

    print(f"\n[Generated Figures]")
    print(f"  results/figures/figure_02_covariance.png")
    print(f"  results/figures/figure_03_eigenvalues.png")
    print(f"  results/figures/figure_05_classical_vs_quantum.png")
    print(f"  results/figures/classical_vs_quantum.png")

if __name__ == "__main__":
    run_beamforming_experiment()
