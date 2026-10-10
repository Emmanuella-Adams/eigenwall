# EigenWall: Research Specification
### Qollab × IonQ Hackathon 2026

**Version:** 1.0  
**Status:** Comprehensive Research Specification  
**Primary theme:** Quantum Impact  
**Core domain:** Quantum computing × adaptive signal processing × deep-space communications

---

## 1. Project Identity

### Project name
# **EigenWall**

### Working subtitle
**Quantum-Assisted Adaptive Beamforming for Deep-Space Communication**

### One-sentence description
> EigenWall investigates whether quantum linear-system techniques can reproduce the covariance-matrix inversion underlying adaptive beamforming, using a simulated deep-space communication scenario as an experimental testbed.

### Core principle
The project **does not assume quantum advantage**.

The investigation experimentally examines:
$\[
\boxed{
\text{Can quantum linear-system methods produce a useful approximation of }
R^{-1}a?
}
\]$
---

# 2. Research Problem

Deep-space spacecraft transmit extremely weak signals over astronomical distances.

An antenna array receives not only the desired spacecraft signal, but also corrupted measurements containing:
- thermal noise,
- external radio-frequency interference (RFI),
- signals arriving from other celestial and terrestrial directions,
- correlated environmental noise,
- receiver front-end noise.

The antenna array must determine the optimal complex weight (amplitude and phase) for each antenna element.

This constitutes the fundamental problem of **adaptive beamforming**.

The research focuses on the Minimum Variance Distortionless Response (MVDR) / Capon formulation.

Given the received signal covariance matrix $\(R\)$ and the target steering vector $\(a\)$, the optimal beamforming weights are given by:

$\[\boxed{w_{\mathrm{MVDR}}
=\frac{R^{-1}a}{a^H R^{-1}a}}\]$

Consequently, the central computational bottleneck is the solution of the linear system:

$\[\boxed{Rw=a}\]$
or equivalently:

$\[
\boxed{
w \propto R^{-1}a.
}
\]$

EigenWall investigates whether quantum linear-system techniques can perform this operation with fidelity sufficient to preserve spatial beamforming characteristics.

---

# 3. Research Motivation

Classical adaptive beamforming solves linear systems efficiently for small and moderate antenna arrays $(\(\mathcal{O}(N^3)\)$ scaling).

However, motivation for investigating quantum methods stems from the mathematical spectral structure of the problem.

The covariance matrix $\(R\)$ admits the eigendecomposition:

$\[R=V\Lambda V^H\]$

where:

$$
\Lambda =
\begin{bmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_N
\end{bmatrix}.
$$

Consequently:

$\[
R^{-1}=V\Lambda^{-1}V^H.
\]$

Therefore:

$\[
R^{-1}a=V\Lambda^{-1}V^Ha.
\]$

The matrix inversion operation is interpreted spectrally as:

 $\[
\lambda_i
\rightarrow
\frac{1}{\lambda_i}.
\]$

This spectral transformation establishes the exact mathematical connection between adaptive beamforming and quantum linear-system algorithms (HHL-style), which estimate eigenvalues in a quantum register and apply controlled rotations inversely proportional to those eigenvalues.

---

# 4. Central Research Question

### Primary question
> **Can a QPE-based quantum linear-system approach approximate the inverse-covariance operation required by adaptive beamforming closely enough to preserve useful beamforming performance for simulated antenna arrays?**

---

# 5. Secondary Research Questions

The research explores seven secondary questions:

### RQ1: Solution accuracy
How closely does the quantum solution approximate the normalized classical solution?

$\[
F = |\langle w_{\mathrm{classical}} | w_{\mathrm{quantum}}\rangle|^2
\]$

### RQ2: Beamforming performance
Does the quantum-derived solution preserve principal radiation pattern characteristics:
- target power gain,
- interference suppression,
- null depth,
- beam pointing direction?

### RQ3: QPE precision
How does the number of phase-estimation qubits \(m\) affect solution fidelity:

$\[
m_{\mathrm{QPE}} \rightarrow F?
\]$

### RQ4: Condition number
How does the condition number $\(\kappa(R) = \lambda_{\max}/\lambda_{\min}\)$ affect quantum solution quality and success probability:

$\[
\kappa(R) \rightarrow F?
\]$

### RQ5: Quantum noise
How does simulated quantum gate and readout noise degrade solution fidelity and beamforming null depth:

$\[
F_{\mathrm{ideal}} \quad \text{vs} \quad F_{\mathrm{noisy}}?
\]$

### RQ6:  Resource requirements
What quantum resources (qubit count, circuit depth, gate count) are required as antenna element count scales?

---

# 6. Research Hypotheses

The investigation formulates five testable scientific hypotheses:

### H1: Solution hypothesis
A QPE-based quantum linear-system procedure can prepare a quantum state whose normalized amplitudes approximate the classical solution $\(R^{-1}a\)$ with high $fidelity (\(F > 0.95\))$ under well-conditioned spectral properties.

### H2: Beamforming hypothesis
If the quantum solution sufficiently approximates the classical solution, the resulting synthesized radiation pattern will preserve the principal characteristics of the classical MVDR beamformer:
- unity gain toward the desired signal direction $(\(w^H a = 1\))$,
- deep attenuation nulls toward interference directions.

### H3: Precision hypothesis
Increasing QPE phase register precision reduces eigenvalue estimation errors within the half-bin bound $\(|\epsilon_\lambda| \le \Delta_\lambda/2\)$, thereby improving the accuracy of the controlled rotation inversion.

### H4: Conditioning hypothesis
Covariance matrices with high condition numbers degrade quantum inversion fidelity and success probability because small eigenvalues $\(\lambda \to 0\)$ amplify phase estimation errors and reduce ancilla rotation success probabilities $\(\mathcal{O}(1/\kappa^2)\)$.

### H5: Noise hypothesis
Physical quantum gate noise (depolarizing and dephasing) introduces amplitude dispersion into the quantum state, degrading null depth and interference rejection.

---

# 7. Classical Mathematical Model

The signal model simulates an antenna array of $\(N\)$ elements receiving signals over $\(T\)$ snapshots.

For snapshot $\(t \in \{1, \dots, T\}\)$:

$\[
x(t) = a(\theta_s)s(t) + \sum_{k=1}^{K} a(\theta_k)i_k(t) + n(t)
\]$

where:
- $\(a(\theta_s) \in \mathbb{C}^N\)$ = steering vector of the desired spacecraft signal,
- $\(s(t) \in \mathbb{C}\)$ = desired signal waveform,
- $\(a(\theta_k) \in \mathbb{C}^N\)$ = steering vector of interference source $\(k\)$,
- $\(i_k(t) \in \mathbb{C}\)$ = interference waveform,
- $\(n(t) \sim \mathcal{CN}(0, \sigma_n^2 I_N)\)$ = additive complex Gaussian thermal noise.

Collecting snapshots into matrix $\(X \in \mathbb{C}^{N \times T} = [x(1), \dots, x(T)]\)$, the sample covariance matrix is:

$\[
R = \frac{1}{T} X X^H.
\]$

To prevent numerical rank deficiency and enhance robustness, diagonal loading is applied:

$\[
R_{\mathrm{loaded}} = R + \epsilon I_N.
\]$

---

# 8. Array Model

The initial configuration models a Uniform Linear Array (ULA):
- Number of antenna elements: $\(N = 4\)$ $(and \(N = 2\)$ for toy benchmarks).
- Inter-element spacing: $\(d = \lambda / 2\)$.
- Wavelength: $\(\lambda = 1.0\)$.

The array steering vector for azimuth angle $\(\theta\)$ is defined by:

$$
a(\theta) =
\begin{bmatrix}
1 \\
e^{j k d \sin\theta} \\
e^{j 2 k d \sin\theta} \\
\vdots \\
e^{j (N-1) k d \sin\theta}
\end{bmatrix},
\qquad
k = \frac{2\pi}{\lambda}.
$$

---

# 9. Deep-Space Operational Scenario

The baseline simulation scenario represents a deep-space communications link:
- **Desired spacecraft signal:** $\(\theta_s = 20^\circ\)$, signal power $\(P_s = 1.0\)$.
- **Interfering source:** $\(\theta_i = -30^\circ\)$, interference-to-noise ratio $\(\mathrm{INR} \approx +6\text{ dB}\)$.
- **Background thermal noise:** variance $\(\sigma_n^2 = 0.8\)$.
- **Snapshot count:** $\(T = 1000\)$.

---

# 10. Classical Reference Baseline

The classical MVDR beamformer provides the analytical ground truth reference:

$\[
w_{\mathrm{classical}} = \frac{R^{-1} a}{a^H R^{-1} a}.
\]$

The normalized spatial power response across angles $\(\theta \in [-90^\circ, 90^\circ]\)$ is:

$\[
B(\theta) = |w^H a(\theta)|^2, \quad B_{\mathrm{dB}}(\theta) = 10 \log_{10}\left(\frac{B(\theta)}{\max_\theta B(\theta)}\right).
\]$

---

# 11. Quantum Algorithm Formulation

The quantum formulation maps the linear system $\(R w = a\)$ to quantum states:

1. **State Preparation:** The steering vector $\(a\)$ is amplitude-encoded into the system register:
   $\[
   |a\rangle = \sum_{i=0}^{N-1} c_i |v_i\rangle, \quad \sum_i |c_i|^2 = 1.
   \]$
2. **Eigenvalue Estimation (QPE):** Evolution under unitary operator $\(U = e^{i R t}\)$ transforms the register:
   $\[
   |a\rangle |0\rangle^{\otimes m} \xrightarrow{\mathrm{QPE}} \sum_i c_i |v_i\rangle |\tilde{\lambda}_i\rangle.
   \]$
3. **Controlled Inversion Rotation:** An ancilla qubit is rotated conditioned on the estimated eigenvalue register:
   $\[
   \sum_i c_i |v_i\rangle |\tilde{\lambda}_i\rangle \left( \sqrt{1 - \frac{C^2}{\tilde{\lambda}_i^2}} |0\rangle + \frac{C}{\tilde{\lambda}_i} |1\rangle \right).
   \]$
4. **Uncomputation:** Inverse QPE uncomputes the phase register back to $\(|0\rangle^{\otimes m}\)$:
   $\[
   \sum_i c_i \frac{C}{\tilde{\lambda}_i} |v_i\rangle |0\rangle^{\otimes m} |1\rangle_{\mathrm{ancilla}} + |\text{trash}\rangle |0\rangle_{\mathrm{ancilla}}.
   \]$
5. **Post-selection:** Measuring the ancilla in state \(|1\rangle\) yields the normalized solution state:
   $\[
   |w_q\rangle \propto R^{-1} |a\rangle.
   \]$
6. **Physical Weight Reconstruction:** Physical distortionless beamforming weights are reconstructed via:
   $\[
   w_{\mathrm{phys}} = \frac{|w_q\rangle}{a^H |w_q\rangle}.
   \]$

---

# 12. Scientific Constraints and Non-Claims

The project **explicitly disclaims unproven quantum speedup**:
- The current scale $(\(N=2, 4\))$ is chosen for precision simulation and trapped-ion execution feasibility, not asymptotic supremacy.
- Standard quantum linear system algorithms assume efficient state preparation, Hamiltonian simulation, and well-behaved condition numbers.
- Full classical weight readout requires quantum state tomography scaling with $\(\mathcal{O}(N)\)$ shots; however, in quantum-native receiver architectures, the state $\(|w\rangle\)$ could interface directly with an RF quantum interferometer without classical readout.

---

# 13. Experimental Matrix

| Experiment | Title | Scientific Objective | Key Outputs |
|---|---|---|---|
| **E1** | Golden Test Benchmark | Validates analytical inversion on 2×2 diagonal and Hermitian systems | Fidelity $\(F\)$, residual error, Figure 1 |
| **E2** | End-to-End Beamforming | Compares classical MVDR and quantum HHL radiation patterns on simulated array | Figure 5, Figure 2, Figure 3, Metrics Dashboard |
| **E3** | QPE Precision Sweep | Measures eigenvalue error and state fidelity as phase register $\(m\)$ varies | Figure 6, precision scaling table |
| **E4** | Condition Number Sweep | Evaluates fidelity and success probability across $\(\kappa \in [1.3, 13.3]\)$ | Figure 7, conditioning table |
| **E5** | Quantum Noise Sweep | Simulates depolarizing error rates $\(p \in [0, 0.04]\)$ on fidelity and null depth | Figure 8, noise robustness table |

---

# 14. Performance Metrics

1. **State Fidelity:** $\(F = |\langle w_c | w_q \rangle|^2\)$
2. **Linear System Residual Error:** $\(\epsilon = \|R w_q - a\|_2 / \|a\|_2\)$
3. **Target Power Gain:** $\(G_s = |w^H a(\theta_s)|^2\)$$
4. **Interference Power Gain:** $\(G_i = |w^H a(\theta_i)|^2\)$
5. **Null Depth:** $\(\mathrm{ND} = 10 \log_{10}(G_s / G_i)\) (dB)$
6. **Peak Beam Error:** $\(\Delta\theta = |\theta_{\mathrm{quantum}}^{\mathrm{peak}} - \theta_{\mathrm{classical}}^{\mathrm{peak}}|\)$
7. **Quantum Success Probability:** $\(P(\mathrm{ancilla}=1)\)$

---

# 15. Research Thesis Statement

> **EigenWall demonstrates the engineering feasibility of using quantum phase-estimation-based linear-system algorithms to approximate inverse-covariance weighting in adaptive beamforming. Operating on simulated deep-space communication scenarios, the architecture verifies that quantum solutions reproduce classical MVDR beam patterns with high fidelity, maintaining deep spatial nulls toward interference and unity gain toward target spacecraft.**
