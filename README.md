# EigenWall: Quantum-Assisted Adaptive Beamforming for Deep-Space Communication

**Domain:** Quantum Linear Algebra × Adaptive Signal Processing × Deep-Space Telecommunications  
**Version:** 1.0 (Research Prototype)

---

## 1. Executive Summary

**EigenWall** is an experimental investigation into whether quantum linear-system algorithms (HHL-style procedures) can reproduce the inverse-covariance operation central to adaptive antenna beamforming. Using a simulated deep-space communication link as an operational testbed, the project evaluates whether a quantum processor can steer spatial radiation nulls toward interfering emitters while preserving distortionless reception toward a distant spacecraft.

[![Architecture diagram of emmanuella-adams/eigenwall](https://gitdiagram.com/emmanuella-adams/eigenwall/diagram.png)](https://gitdiagram.com/emmanuella-adams/eigenwall?utm_source=readme&utm_medium=picture)

```text
                                 EIGENWALL COMPUTATIONAL PIPELINE
                                 
  [Deep-Space Spacecraft]      [Interference Sources]      [Thermal Noise]
             │                           │                        │
             └───────────────────────────┼────────────────────────┘
                                         ▼
                      ┌─────────────────────────────────────┐
                      │    Uniform Linear Array (ULA)       │
                      │    Snapshot Simulation: X(t)        │
                      └──────────────────┬──────────────────┘
                                         │
                                         ▼
                      ┌─────────────────────────────────────┐
                      │    Sample Covariance Matrix R       │
                      │    Diagonal Loading & Scaling       │
                      └──────────────────┬──────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         [Classical MVDR Path]                     [Quantum Inversion Path]
         Solves Rw = a                             Encodes |a> on System Qubits
         w_c = R^-1 a / (a^H R^-1 a)               Quantum Phase Estimation (QPE)
                    │                              Controlled Inversion Rotation
                    │                              Uncomputation & Post-Selection
                    │                              w_q = |w_q> / (a^H |w_q>)
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                      ┌─────────────────────────────────────┐
                      │   Array Beam Pattern Synthesis      │
                      │   B(theta) = |w^H a(theta)|^2       │
                      │   Fidelity & Null Depth Comparison  │
                      └─────────────────────────────────────┘
```

---

## 2. Deep-Space Communication Context & Motivation

### The Operational Challenge
Spacecraft traversing interplanetary space (e.g., Mars missions, Europa Clipper, Voyager 1/2) transmit radio signals over tens of millions of kilometers. By the time these wavefronts reach Earth:
- **Extreme Path Loss:** Received signal power is astronomically low (often below $-160\text{ dBm}$, with signal-to-noise ratios $\text{SNR} < -10\text{ dB}$).
- **Dense Radio Frequency Interference (RFI):** Ground stations (such as NASA's Deep Space Network in Goldstone, Madrid, and Canberra) operate in an increasingly congested spectrum contaminated by Low Earth Orbit (LEO) mega-constellations, terrestrial radar, and adjacent commercial satellite uplinks.
- **The Phased Array Solution:** To overcome path loss and interference, ground receivers combine multiple antenna elements into phased arrays, adjusting the complex weights (amplitude and phase) of each antenna element to synthesize a high-gain pencil beam pointing at the spacecraft while steering sharp attenuation nulls directly toward interference sources.

### The Computational Bottleneck
The standard optimum spatial filter is the **Minimum Variance Distortionless Response (MVDR) / Capon beamformer**:
$$w_{\mathrm{MVDR}} = \frac{R^{-1} a}{a^H R^{-1} a}$$

where $R \in \mathbb{C}^{N \times N}$ is the array covariance matrix and $a \in \mathbb{C}^N$ is the steering vector of the target spacecraft.

Computing $R^{-1} a$ classically requires:
- $\mathcal{O}(N^3)$ operations via direct LU/Cholesky factorization.
- $\mathcal{O}(N^2)$ operations per snapshot via recursive least squares (RLS).

As deep-space receiver arrays scale from tens to hundreds or thousands of elements ($N = 64, 256, 1024$) to achieve extreme aperture gains, the computational latency of classical covariance matrix inversion outpaces the channel coherence time.

### Potential Quantum Advantage
Quantum linear system algorithms (the Harrow-Hassidim-Lloyd / HHL framework and modern Quantum Singular Value Transformation / QSVT) solve linear systems of equations $R w = a$ in time scaling:
$$\mathcal{O}\left( \text{polylog}(N) \cdot s^2 \kappa^2 / \epsilon \right)$$
where $N$ is matrix dimension, $s$ is matrix sparsity, $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number, and $\epsilon$ is target error.

For massive antenna arrays ($N \gg 10^3$), quantum linear solvers offer an exponential reduction in vector space dimension handling, encoding $N$-element arrays into $n = \lceil \log_2 N \rceil$ qubits.

---

## 3. The Trapped-Ion (IonQ) Hardware Rationale

EigenWall is built to run on the **IonQ Trapped-Ion Simulator and Hardware architecture**. Trapped-ion systems offer specific structural advantages for quantum adaptive beamforming:

1. **All-to-All Qubit Connectivity:**
   - In QPE and HHL, multi-controlled unitary evolutions and inversion rotations require interactions between distant clock qubits and system qubits.
   - On nearest-neighbor superconducting grid architectures, multi-qubit gates incur heavy SWAP-gate overhead, increasing circuit depth by factors of $5\times$ to $10\times$.
   - IonQ's trapped $^{171}\text{Yb}^+$ or $^{133}\text{Ba}^+$ ions communicate via shared collective motional modes in a linear crystal, enabling direct two-qubit entangling gates between any arbitrary pair of qubits without routing overhead.
2. **Native Two-Qubit Gate Set:**
   - IonQ natively executes the Mølmer-Sørensen gate $R_{XX}(\theta) = \exp\left(-i \frac{\theta}{2} X \otimes X\right)$, which synthesizes multi-controlled rotations and quantum Fourier transforms with high operational fidelity.
3. **Low Idle Dephasing:**
   - Trapped-ion qubits possess hyperfine ground-state coherence times exceeding seconds ($T_2 > 1\text{ s}$), minimizing dephasing errors during multi-stage QPE uncomputation.

---

## 4. Scientific Integrity & Boundary Constraints

EigenWall adheres to strict technical rigor:

> **I do not make unproven claims of immediate quantum supremacy.**

Quantum linear system algorithms involve known real-world engineering constraints that EigenWall explicitly models:
1. **The State Preparation Bottleneck:** Initializing arbitrary classical steering vectors $|a\rangle$ into a quantum register generally requires $\mathcal{O}(N)$ gates unless structured isometric initialization or quantum RAM (QRAM) is available.
2. **The Condition Number Bottleneck:** If $\kappa(R) \gg 1$, the post-selection success probability $P(\text{ancilla}=1) = \mathcal{O}(1/\kappa^2)$ drops significantly, requiring amplitude amplification. EigenWall mitigates this via adaptive diagonal loading (regularization).
3. **The State Readout Bottleneck:** Reconstructing the full classical weight vector $w$ from the quantum state $|w\rangle$ requires quantum state tomography scaling with $\mathcal{O}(N)$ shots. In future quantum-native RF architectures, however, the state $|w\rangle$ could modulate incoming photonics directly without intermediate classical readout.

---

## 5. Mathematical Pipeline & Algorithm

### Step 1: Physical Signal Model
Simulates snapshot vector $x(t) \in \mathbb{C}^N$:
$$x(t) = a(\theta_s) s(t) + \sum_{k=1}^K a(\theta_k) i_k(t) + n(t)$$
where $a_n(\theta) = \exp(j n k d \sin\theta)$, $k = 2\pi/\lambda$, and $d = \lambda/2$.

### Step 2: Covariance Matrix Regularization & Scaling
Sample covariance $R = \frac{1}{T} X X^H$ is regularized via diagonal loading:
$$R_{\mathrm{loaded}} = R + \epsilon I$$
To ensure that all eigenphases map into the unambiguous phase interval $[0, 0.5)$ of the QPE register without aliasing, $R$ is scaled:
$$R_{\mathrm{scaled}} = R_{\mathrm{loaded}} \cdot \left(\frac{2.0}{\lambda_{\max}}\right)$$
Because MVDR weights are scale-invariant, the exact beamforming solution is preserved.

### Step 3: Quantum Linear Inversion (HHL)
1. **State Preparation:** Steering vector $a$ is normalized and initialized on system register:
   $$|a\rangle = \sum_{i} c_i |v_i\rangle$$
2. **Quantum Phase Estimation:** Evolution under $U = \exp(i R_{\mathrm{scaled}} t)$ with $t = \pi / \lambda_{\max}$ estimates eigenphases:
   $$\sum_i c_i |v_i\rangle |0\rangle^{\otimes m} \xrightarrow{\mathrm{QPE}} \sum_i c_i |v_i\rangle |\tilde{\lambda}_i\rangle$$
3. **Controlled Inversion Rotation:** Multi-controlled $R_y(\theta_j)$ rotates the ancilla qubit:

$$
\theta_j = 2 \arcsin\left(\min\left(1.0, \frac{C}{\tilde{\lambda}_j}\right)\right),
\qquad C \le \lambda_{\min}
$$
4. **Uncomputation:** Inverse QPE uncomputes the clock register back to $|0\rangle^{\otimes m}$.
5. **Post-Selection:** Projecting the ancilla into state $|1\rangle$ yields:
   $$|w_q\rangle \propto R^{-1} |a\rangle$$

### Step 4: Physical Weight Reconstruction
Physical weights satisfying the distortionless constraint ($w_{\mathrm{phys}}^H a = 1$) are reconstructed via:
$$w_{\mathrm{phys}} = \frac{|w_q\rangle}{a^H |w_q\rangle}$$
which simultaneously removes arbitrary global phase factors.

---

## 6. Experimental Suite & Empirical Results

### Summary of Performance Dashboard ($N=4$, Target: $20^\circ$, Interferer: $-30^\circ$)
From [`results/tables/metrics_dashboard.csv`](file:///c:/Users/Lenovo/Desktop/eigenwall/results/tables/metrics_dashboard.csv):

| Metric | Classical MVDR Baseline | Quantum HHL Solution | Delta / Error |
|---|---:|---:|---:|
| **Solution Fidelity $F$** | `1.0000` | **`1.0000`** | `< 0.0001` |
| **Target Power Gain ($20^\circ$)** | `1.0000` | **`1.0000`** | `0.0000` |
| **Interference Power Gain ($-30^\circ$)** | `6.31e-4` | **`7.89e-4`** | `1.58e-4` |
| **Interference Null Depth** | `32.00 dB` | **`31.03 dB`** | **`0.97 dB`** |
| **Peak Pointing Direction** | `20.8°` | **`20.8°`** | **`0.00°`** |
| **Linear System Residual Error** | `0.0000` | **`0.0076`** | `0.0076` |
| **Quantum Circuit Qubits** | — | **`7`** | — |
| **Circuit Depth** | — | **`52`** | — |

---

## 7. Generated Figures & Experimental Outputs

All generated figures are saved under [`results/figures/`](file:///c:/Users/Lenovo/Desktop/eigenwall/results/figures/):

| Figure File | Description | Core Scientific Takeaway |
|---|---|---|
| [`figure_01_toy_inversion.png`](file:///c:/Users/Lenovo/Desktop/eigenwall/results/figures/figure_01_toy_inversion.png) | Golden Tests A & B inversion | Validates exact quantum inversion against analytical solutions ($F \ge 0.999$). |
| [`figure_02_covariance.png`](file:///c:/Users/Lenovo/Desktop/eigenwall/results/figures/figure_02_covariance.png) | Covariance magnitude and phase | Confirms Hermiticity and non-diagonal correlation structure of the antenna array. |
| [`figure_03_eigenvalues.png`](file:///c:/Users/Lenovo/Desktop/eigenwall/results/figures/figure_03_eigenvalues.png) | Covariance eigenvalue spectrum | Displays condition number $\kappa = 8.46$ matching the QPE dynamical window. |
| [`figure_05_classical_vs_quantum.png`](file:///c:/Users/Lenovo/Desktop/eigenwall/results/figures/figure_05_classical_vs_quantum.png) | Classical vs Quantum Beam Pattern | **Primary Deliverable:** Demonstrates $>31\text{ dB}$ null depth and exact pointing alignment. |
| [`figure_06_qpe_precision.png`](file:///c:/Users/Lenovo/Desktop/eigenwall/results/figures/figure_06_qpe_precision.png) | QPE precision sweep ($m \in [2, 5]$) | Quantifies eigenvalue estimation error convergence within theoretical bounds. |
| [`figure_07_condition_number.png`](file:///c:/Users/Lenovo/Desktop/eigenwall/results/figures/figure_07_condition_number.png) | Sensitivity vs condition number $\kappa$ | Confirms that high $\kappa$ reduces success probability $P(\text{succ})$ and degrades inversion. |
| [`figure_08_noise_performance.png`](file:///c:/Users/Lenovo/Desktop/eigenwall/results/figures/figure_08_noise_performance.png) | Depolarizing noise robustness | Demonstrates that practical null depth remains $>23\text{ dB}$ up to $p = 4\%$ gate error. |

---

## 8. Installation, Execution & IonQ Integration

### Prerequisites
- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Automated Unit Testing
Run the complete test suite (9 unit tests validating signal models, covariance matrices, classical solvers, and quantum circuits):
```bash
python run_tests.py
# Or with pytest
python -m pytest tests -v
```

### Master Runner with IonQ Simulator Support
The script [`first.py`](file:///c:/Users/Lenovo/Desktop/eigenwall/first.py) executes the full end-to-end pipeline:
```bash
python first.py
```
- **IonQ Cloud Execution:** If an `IONQ_API_KEY` is detected in your environment or `.env` file, `first.py` automatically authenticates with `qiskit_ionq.IonQProvider` and executes on the IonQ Cloud Simulator (`simulator`).
- **Trapped-Ion Emulation:** If no API key is set, `first.py` transpiles the circuit into IonQ native trapped-ion basis gates (`rxx`, `rx`, `ry`, `rz`, `h`) and simulates the hardware execution on `AerSimulator`.
- *This effort is supported by Qollab & IonQ*

### Reproducing All Individual Experiments
```bash
python experiments/experiment_01_toy.py
python experiments/experiment_02_beamforming.py
python experiments/experiment_03_qpe_precision.py
python experiments/experiment_04_condition_number.py
python experiments/experiment_05_noise.py
```

### Development and Verification

Parts of the codebase were initially developed with assistance from Google Antigravity. I subsequently tested and verified the implementation locally, checking its execution and behavior within the available development environment.
