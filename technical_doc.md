# EigenWall: Technical Specification & Architecture Document

### Qollab × IonQ Hackathon 2026

**Version:** 1.0  
**Status:** Complete Technical Build Specification  
**Architecture:** Quantum-Assisted Adaptive Signal Processing (MVDR Beamforming)

---

# 1. Technical Objective

EigenWall constructs a reproducible computational prototype executing the end-to-end signal processing and quantum linear algebra pipeline:

$$
\boxed{
\begin{array}{c}
\text{Physical Signal Simulation} \\
\Downarrow \\
\text{Sample Covariance Matrix } R \\
\Downarrow \\
\text{Linear System Formulation } Rw = a \\
\Downarrow \\
\text{Classical MVDR Baseline Solution } w_c \\
\Downarrow \\
\text{Quantum Linear Inversion (QPE + Controlled Rotation + Uncomputation)} \\
\Downarrow \\
\text{Physical Weight Reconstruction } w_q \\
\Downarrow \\
\text{Spatial Beam Pattern Synthesis and Validation}
\end{array}
}
$$

The engineering prototype focuses on small antenna arrays ($N \in \{2,4\}$) to enable rigorous verification against analytical solutions, exact statevector benchmarking, and execution on the IonQ trapped-ion simulator.

---

# 2. Technology Stack & Dependencies

| Component | Specification | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core development language |
| **Quantum Framework** | Qiskit 2.3+ | Quantum circuit synthesis, QPE, unitary synthesis |
| **Quantum Simulation** | Qiskit Aer 0.13+ | Ideal statevector simulation and noisy depolarizing channel simulation |
| **IonQ Provider** | `qiskit-ionq` 1.1+ | Trapped-ion backend execution on IonQ Cloud Simulator / Hardware |
| **Numerical Linear Algebra** | NumPy 1.24+ | Vectorized snapshot generation, matrix operations |
| **Scientific Algorithms** | SciPy 1.10+ | Matrix exponentials ($e^{iRt}$), eigendecompositions |
| **Data Handling** | Pandas 2.0+ | Structured metrics dashboards, tabular logging |
| **Visualization** | Matplotlib 3.7+ | Radiation patterns, covariance heatmaps, spectra |
| **Testing** | Pytest 7.0+ | Unit and integration regression testing |

---

# 3. Repository Architecture

```text
EigenWall/
│
├── README.md                       # Comprehensive project documentation
├── LICENSE                         # Open-source MIT license
├── requirements.txt                # Pinned package dependencies
├── run_tests.py                    # Self-contained automated test runner
├── first.py                         # Self-contained master execution script with IonQ support
│
├── eigenwall/                       # Core computational library
│   ├── __init__.py                  # Package root
│   ├── signal_model.py              # ULA geometry, steering vectors, snapshot generator
│   ├── covariance.py                # Sample covariance estimation & diagonal loading
│   ├── beamforming.py               # Classical MVDR, beam patterns, quantum weight mapping
│   ├── matrix_utils.py              # State normalization, Hermiticity, condition checks
│   ├── metrics.py                   # Fidelity, null depth, angular error, residual norms
│   │
│   └── quantum/                     # Quantum subroutines
│       ├── __init__.py              # Quantum subpackage initialization
│       ├── state_preparation.py     # Amplitude encoding into quantum registers
│       ├── qpe.py                   # Quantum Phase Estimation & bit-exact phase decoder
│       ├── hhl_solver.py            # Complete HHL solver with uncomputation
│       └── measurement.py           # Ancilla post-selection & empirical readout
│
├── experiments/                     # Automated scientific experiment suite
│   ├── __init__.py
│   ├── experiment_01_toy.py         # Golden Tests A & B (analytical matrix inversions)
│   ├── experiment_02_beamforming.py # End-to-end MVDR beamforming benchmark (Figure 5)
│   ├── experiment_03_qpe_precision.py # Register precision sweep (m in {2, 3, 4, 5})
│   ├── experiment_04_condition_number.py # Numerical conditioning analysis
│   └── experiment_05_noise.py       # Depolarizing noise robustness analysis
│
├── notebooks/                       # Interactive exploration notebooks
│   ├── 01_signal_model.ipynb
│   ├── 02_covariance_and_mvdr.ipynb
│   ├── 03_toy_qpe.ipynb
│   ├── 04_quantum_linear_solver.ipynb
│   ├── 05_quantum_beamforming.ipynb
│   └── 06_noise_and_scaling.ipynb
│
├── results/                         # Generated experimental outputs
│   ├── figures/                     # High-resolution publication plots (Figures 1 to 8)
│   └── tables/                      # CSV performance dashboards
│
└── tests/                           # Test suite
    ├── __init__.py
    ├── test_signal_model.py
    ├── test_covariance.py
    ├── test_classical_solver.py
    └── test_quantum_solver.py
```

---

# 4. Mathematical & Algorithmic Modules

## 4.1 Signal Model (`signal_model.py`)

Models a Uniform Linear Array (ULA) of $N$ isotropic antenna elements:

- **Inter-element spacing:** $d = \lambda/2$.
- **Wavelength:** $\lambda = 1.0$.
- **Element indices:** $n \in \{0,1,\dots,N-1\}$.

### Steering Vector

The steering vector for an arrival angle $\theta$ is defined as

$\[
a_n(\theta)=\exp\left(
j \cdot n \cdot \frac{2\pi}{\lambda}
\cdot d \cdot \sin(\theta)
\right).
\]$

### Signal Synthesis

Signal synthesis over $T$ snapshots is defined as

$\[
x(t)=a(\theta_s)s(t)
+
\sum_{k=1}^{K}a(\theta_k)i_k(t)
+
n(t),
\]$

where the additive noise follows the complex Gaussian distribution

$$
n(t)\sim\mathcal{CN}(0,\sigma^2 I_N).
$$

Here, $\theta_s$ is the desired signal's direction of arrival, $\theta_k$ denotes the direction of arrival of the $k$-th interferer, and $\sigma^2$ is the noise variance.

## 4.2 Covariance Analysis (`covariance.py`)

Computes the sample covariance matrix:

$$
R=\frac{1}{T}XX^H,
\qquad
X\in\mathbb{C}^{N\times T}.
$$

Applies diagonal loading (regularization) to guarantee strict positive definiteness when $\epsilon>0$:

$\[
R_{\mathrm{loaded}}=R+\epsilon I_N.
\]$

To prevent QPE phase aliasing, spectral scaling is applied:

$\[
R_{\mathrm{scaled}}=R_{\mathrm{loaded}}
\cdot
\left(
\frac{2.0}
{\lambda_{\max}(R_{\mathrm{loaded}})}
\right).
\]$

Because classical MVDR beamforming weights are scale-invariant,

$\[
\frac{(cR)^{-1}a}{a^H(cR)^{-1}a}=\frac{R^{-1}a}{a^H R^{-1}a},
\qquad c>0,
\]$

this scaling preserves the exact physical solution while mapping all eigenvalues to $(0,2.0]$, assuming the loaded covariance matrix is positive definite.

## 4.3 Classical MVDR Beamforming (`beamforming.py`)

Calculates the distortionless MVDR weight vector:

$\[
w=\frac{R^{-1}a}{a^H R^{-1}a},
\]$

guaranteeing the distortionless constraint

$\[
w^H a=1.
\]$

Calculates the spatial array power response:

$[
B(\theta)=|w^H a(\theta)|^2.
\]$

The normalized decibel response is

$\[
B_{\mathrm{dB}}(\theta)=10\log_{10}
\left(
\frac{B(\theta)}{\max_{\theta}B(\theta)}
\right).
\]$

### Quantum Weight Reconstruction

Reconstructs physical beamforming weights from a normalized quantum solution state $|w_q\rangle$:

$$
w_{\mathrm{phys}}=\frac{|w_q\rangle}{a^H|w_q\rangle}=frac{|w_q\rangle}{\mathrm{vdot}(a,|w_q\rangle)}.
$$

This normalization removes global amplitude and phase ambiguity while enforcing the distortionless constraint

$$
w_{\mathrm{phys}}^H a=1.
$$

The denominator must be nonzero for the reconstruction to be defined.

---

# 5. Quantum Subroutine Architecture

## 5.1 Register Allocation

The quantum linear solver operates across three dedicated quantum registers:

1. **System Register (`sys`):** $n_s=\lceil\log_2 N\rceil$ qubits encoding the target vector $|a\rangle$.
2. **Phase Register (`phase`):** $m$ clock qubits for Quantum Phase Estimation.
3. **Ancilla Register (`ancilla`):** One qubit flagging successful eigenvalue inversion upon post-selection into state $|1\rangle$.

### Total Qubit Requirement

$$
n_{\mathrm{total}}=n_s+m+1.
$$

For $N=4$ and $m=4$:

$$
n_{\mathrm{total}}=2+4+1=7\text{ qubits}.
$$

## 5.2 Algorithmic Execution Steps

```text
|0>_ancilla ─────────────── [Ry(theta)] ─────────────── M(ancilla=1)
                                │
|0>_phase   ──[H^m]─── [IQFT] ──*─────── [QFT] ──[H^m]─
                         │                 │
|b>_sys     ────────── [U^(2^k)] ─────── [U^-(2^k)] ─── Solution State |w>
```

1. **State Preparation:** The normalized steering vector $|a\rangle$ is prepared on `sys` using isometric amplitude initialization.

2. **Phase Superposition:** Hadamard gates $H^{\otimes m}$ create equal superposition across `phase`.

3. **Controlled Unitary Evolution:** Controlled unitaries $C\text{-}U^{2^k}$, where $U=\exp(iRt)$, are applied sequentially with evolution time

   $\[
   t=\frac{\pi}{\lambda_{\max}}.
   \]$

4. **Inverse QFT:** The phase register is transformed using `QFTGate(m).inverse()`.

5. **Phase Decoding:** The measured integer register value $j=\mathrm{int}(\mathrm{bits},2)$ corresponds to phase

   $\[
   \phi_j=\frac{j}{2^m}
   \]$

   and eigenvalue estimate

   $\[
   \widetilde{\lambda}_j=\frac{2\pi\phi_j}{t}.
   \]$

6. **Controlled Inversion Rotation:** Multi-controlled $R_y(\theta_j)$ gates target `ancilla`, conditioned on `phase = j`:

   $\[
   \theta_j=2\arcsin\left(
   \min\left(1.0,\frac{C}{\widetilde{\lambda}_j}\right)
   \right),
   \qquad
   C\leq\lambda_{\min}.
   \]$

8. **Uncomputation:** Forward `QFTGate(m)` and inverse controlled unitaries $C\text{-}U^{-2^k}$ disentangle the phase register back to $|0\rangle^{\otimes m}$.

9. **Post-selection:** Conditioning on $\text{ancilla}=|1\rangle$ yields the normalized solution state

   $\[
   |w_q\rangle\propto R^{-1}|a\rangle.
   \]$

---

# 6. IonQ Hardware & Simulator Execution Architecture

The system provides execution paths for IonQ trapped-ion hardware and simulators:

```text
               ┌───────────────────────────────────────┐
               │    EigenWall HHL Quantum Circuit      │
               └───────────────────┬───────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   [IonQ API Key Present]                    [Local Offline Execution]
              │                                         │
              ▼                                         ▼
   qiskit_ionq.IonQProvider                 Qiskit AerSimulator
              │                                         │
              ▼                                         ▼
   provider.get_backend('simulator')        Transpilation to IonQ Basis
              │                             ['rxx', 'rx', 'ry', 'rz', 'h']
              ▼                                         │
   IonQ Cloud Trapped-Ion Simulator                     ▼
   (All-to-all native connectivity)         Trapped-Ion Emulation Mode
```

### Trapped-Ion Architecture Advantages

1. **All-to-All Connectivity:** Reduces SWAP-gate overhead associated with nearest-neighbor superconducting layouts when performing multi-controlled phase operations.

2. **Native Gate Set:** Native Mølmer–Sørensen $R_{XX}(\theta)$ two-qubit entangling gates support high-fidelity multi-qubit unitary synthesis.

3. **Execution Support:** Automated via `first.py` with environment detection of `IONQ_API_KEY`.

---

# 7. Verification Protocols & Acceptance Criteria

## Golden Tests

### 1. Golden Test A (Diagonal)

$$
A=\text{diag}(1,2),
\qquad
b=\begin{bmatrix}
1\\
1
\end{bmatrix}.
$$

Analytical solution:

$$
x=\begin{bmatrix}
1\\
0.5
\end{bmatrix}.
$$

Acceptance threshold: $F\geq0.99$.

**Reported result:** $F=1.00000$.

### 2. Golden Test B (Hermitian)

$$
A=\begin{bmatrix}
1&0.2\\
0.2&2
\end{bmatrix},
\qquad
b=\begin{bmatrix}
1\\
1
\end{bmatrix}.
$$

Acceptance threshold: $F\geq0.95$.

**Reported result:** $F=0.99899$.

### 3. Covariance Beamforming

- Array size: $N=4$.
- Desired signal direction: $\theta_s=20^\circ$.
- Interference direction: $\theta_i=-30^\circ$.

Acceptance criteria:

- Null depth $\geq25\,\text{dB}$.
- Beam direction error $\leq1.0^\circ$.

**Reported results:**

- Null depth: $31.03\,\text{dB}$.
- Beam direction error: $0.00^\circ$.

---

# 8. Experimental Suite

- **Experiment 01 (`experiment_01_toy.py`):** Golden Test analytical benchmarks. Generates Figure 1 and the results table.

- **Experiment 02 (`experiment_02_beamforming.py`):** End-to-end adaptive beamforming benchmark. Generates Figure 2 (Covariance), Figure 3 (Eigenvalues), and Figure 5 (Classical vs. Quantum Beam Pattern), plus the metrics dashboard.

- **Experiment 03 (`experiment_03_qpe_precision.py`):** Precision scaling across $m\in\{2,3,4,5\}$. Generates Figure 6.

- **Experiment 04 (`experiment_04_condition_number.py`):** Conditioning sweep across $\kappa\in[1.3,13.3]$. Generates Figure 7.

- **Experiment 05 (`experiment_05_noise.py`):** Depolarizing gate error analysis across $p\in[0,0.04]$. Generates Figure 8.
