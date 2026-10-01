# NekWave Convergence Benchmark: $h$- and $p$-Refinement

This benchmark verifies the high-order spatial discretization accuracy of NekWave's Discontinuous Galerkin Spectral Element (DG-SEM) solver on 3D Maxwell curl equations over **10 complete wave periods**.

---

## 1. Physics & Exact Analytical Solution

The benchmark simulates a smooth transverse electromagnetic (TEM) plane wave propagating through a 3D periodic domain $\Omega = [-1, 1] \times [-1, 1] \times [-0.5, 0.5]$:

$$
\mathbf{E}(\mathbf{x}, t) = \begin{pmatrix} 0 \\ 0 \\ \cos(\mathbf{k} \cdot \mathbf{x} - \omega t) \end{pmatrix}, \qquad
\mathbf{H}(\mathbf{x}, t) = \frac{1}{\sqrt{2}} \begin{pmatrix} \cos(\mathbf{k} \cdot \mathbf{x} - \omega t) \\ -\cos(\mathbf{k} \cdot \mathbf{x} - \omega t) \\ 0 \end{pmatrix}
$$

* **Wave vector:** $\mathbf{k} = (\pi, \pi, 0)$, $|\mathbf{k}| = \sqrt{2}\pi$
* **Angular frequency:** $\omega = c |\mathbf{k}| = \sqrt{2}\pi$ ($c = 1$)
* **Exact temporal period:** $T_{\text{period}} = \frac{2\pi}{\omega} = \sqrt{2}$
* **Simulation duration:** $T_{\text{final}} = 10 \times T_{\text{period}} = 10\sqrt{2} \approx 14.1421$

At time $t = T_{\text{final}}$, the wave phase advances by $\omega T_{\text{final}} = 20\pi = 10 \times 2\pi$, so the exact solution returns **identically to the initial state**:
$$
\mathbf{E}(\mathbf{x}, T_{\text{final}}) \equiv \mathbf{E}(\mathbf{x}, 0), \qquad \mathbf{H}(\mathbf{x}, T_{\text{final}}) \equiv \mathbf{H}(\mathbf{x}, 0)
$$

---

## 2. Convergence Studies

### Part 1: $h$-Refinement (Algebraic Convergence)
* **Orders tested:**
  * **$N = 3$** (Degree $P = 2$) $\longrightarrow$ Expected slope $\mathcal{O}(h^3)$
  * **$N = 5$** (Degree $P = 4$) $\longrightarrow$ Expected slope $\mathcal{O}(h^5)$
  * **$N = 8$** (Degree $P = 7$) $\longrightarrow$ Expected slope $\mathcal{O}(h^8)$
* **Mesh sequence:** $K \times K \times 1$ elements with $K \in \{2, 4, 8, 16, 32, 64\}$, so $h = 2/K \in \{1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125\}$.

### Part 2: $p$-Refinement (Spectral / Exponential Convergence)
* **Meshes tested:**
  * **Coarse mesh:** $4 \times 4 \times 1$ hex elements
  * **Refined mesh:** $16 \times 16 \times 1$ hex elements
* **Order sequence:** $N \in \{2, 3, 4, 5, 6, 7, 8\}$ (polynomial degrees $P = N - 1 \in \{1, 2, 3, 4, 5, 6, 7\}$).
* **Expected decay:** Exponential error decay $\mathcal{O}(e^{-\gamma N})$ down to the time-integration / machine precision floor.

### Part 3: Numerical Flux Parameter Scan
The benchmark scans 3 values of $C_0$:
* **$C_0 = 0.0$:** Central Flux (energy-conserving, zero dissipation)
* **$C_0 = 0.5$:** Intermediate Flux
* **$C_0 = 1.0$:** Upwind Flux (strictly dissipative)

---

## 3. How to Build and Run

### Build
From your CMake build directory:
```bash
cmake --build . --target convergence_test
```

### Run via Slurm (LUMI-G)
Submit the dedicated 1-GCD job script:
```bash
sbatch conv.slurm
```

### Output Figures
The script generates 3 figures (each with 2 subplots and 2 lines) in `output/`:
* `convergence_study_C0_0.0.png` (+ `.pdf`)
* `convergence_study_C0_0.5.png` (+ `.pdf`)
* `convergence_study_C0_1.0.png` (+ `.pdf`)
