# Fully-Discrete Analytical Spatio-Temporal Dispersion Analysis

This tool computes the exact theoretical spatio-temporal numerical dispersion and dissipation of the NekWave DGTD solver by coupling the spatial Bloch-Floquet amplification operator with the Low-Storage Runge-Kutta (LSRK45) temporal stability polynomial.

---

## 1. Theoretical Formulation

A complete numerical simulation accumulates errors from two distinct sources:
1. **Spatial Discretization:** Discontinuous Galerkin Spectral Element Method (DG-SEM) with polynomial order $N$ and numerical flux parameter $C_0$ ($0.0 = \text{Central}, 1.0 = \text{Upwind}$).
2. **Temporal Discretization:** Explicit 5-stage, 4th-order Low-Storage Runge-Kutta scheme (Carpenter & Kennedy 1994).

### Spatial Bloch-Floquet Amplification Matrix
Applying the Bloch-Floquet plane wave ansatz $\mathbf{U}_k = \mathbf{U}_0 e^{-j (\mathbf{k} \cdot \Delta\mathbf{r}_k)}$ to a reference element yields the semi-discrete autonomous linear system:

$$\frac{d\mathbf{U}_0}{dt} = \mathbf{H}(\mathbf{k}) \mathbf{U}_0, \qquad \mathbf{H}(\mathbf{k}) = \mathbf{M}^{-1} \left[ \mathbf{S} + \sum_k \mathbf{F}_k e^{-j (\mathbf{k} \cdot \Delta\mathbf{r}_k)} \right]$$

where $\mathbf{M}$ is the diagonal mass matrix, $\mathbf{S}$ is the local volumetric and internal flux stiffness matrix, and $\mathbf{F}_k$ are interface numerical flux matrices coupling the reference element to its neighbors across offset vectors $\Delta\mathbf{r}_k$.

### Fully Discrete Spatio-Temporal Amplification Factor
Integrating the linear ODE $\frac{d\mathbf{U}_0}{dt} = \mathbf{H}(\mathbf{k}) \mathbf{U}_0$ over one time step $\Delta t$ using explicit Runge-Kutta advances the state via the stability polynomial $G(z)$:

$$\mathbf{U}^{n+1} = G\big(\mathbf{H}(\mathbf{k}) \Delta t\big) \mathbf{U}^n$$

For the 5-stage Low-Storage RK45 scheme with Butcher coefficients $(a_i, b_i)$:
$$G(z) = 1 + z + \frac{z^2}{2!} + \frac{z^3}{3!} + \frac{z^4}{4!} + c_5 z^5$$

For any spatial eigenvalue $\lambda \in \operatorname{spec}(\mathbf{H}(\mathbf{k}))$, the scaled complex parameter is $z = \lambda \Delta t$. The fully discrete temporal eigenvalue is:

$$\mu = G(\lambda \Delta t) = |\mu| \, e^{-j \omega_{\text{discrete}} \Delta t}$$

* **Stability / Dissipation:** $|\mu| = |G(z)| \le 1$ ensures numerical stability.
* **Discrete Numerical Frequency:**
  $$\omega_{\text{discrete}} = \frac{\operatorname{arg}(G(z))}{\Delta t}$$
* **Normalized Phase Velocity:**
  $$\frac{v_p}{c} = \frac{\omega_{\text{discrete}}}{c |\mathbf{k}|}$$

---

## 2. Physical Mode Selection & Eigenvector Tracking

The reference element operator $\mathbf{H}(\mathbf{k})$ has $3 N^2$ eigenvalues corresponding to the 3 field components ($E_z, H_x, Hy$) across $N \times N$ GLL nodes. The majority of these represent non-propagating or spurious high-order numerical modes.

To isolate the genuine propagating physical wave without branch hops:
1. **Low-$k$ Initialization:** At $k_{\min} = 0.01$, the physical forward-propagating mode is identified by finding the eigenvalue that minimizes frequency error and damping:
   $$\text{idx}_0 = \arg\min_i \left( |\omega_i - c |\mathbf{k}|| + |\operatorname{Re}(\lambda_i)| \right)$$
2. **Eigenvector Correlation Tracking:** For subsequent wavenumbers $k$, the physical mode is tracked continuously by maximizing the eigenvector inner product with the preceding state:
   $$\text{idx}_{k} = \arg\max_j \frac{|\mathbf{v}_j^\dagger \mathbf{v}_{\text{prev}}|}{\|\mathbf{v}_j\| \|\mathbf{v}_{\text{prev}}\|}$$
   This completely prevents mode-hopping and spurious branch transitions across the entire wavenumber spectrum.

---

## 3. Time Step & CFL Calculation

The time step $\Delta t$ is computed automatically to match NekWave's core solver:
$$\text{CFL}_{\text{auto}} = \text{safetyFactor} \cdot \left(\frac{\alpha_{\text{RK}}}{\sqrt{3}}\right), \qquad \Delta t = \frac{\text{CFL}_{\text{auto}} \Delta x_{\min}}{c}$$
where $\alpha_{\text{RK}} \approx 1.75$ is the 1D stability limit of LSRK45, $\sqrt{3}$ accounts for 3D stability, and $\Delta x_{\min}$ is the minimum distance between Gauss-Lobatto-Legendre (GLL) quadrature nodes. Default `safetyFactor = 0.4`.

---

## 4. Pipeline & Usage

### 1. Compile Matrix Exporter
```bash
cmake --build build -j4
```

### 2. Single Run (Export & Plot)
```bash
# Export reference element matrices for Order N = 6, Central Flux C0 = 0.0
./build/ctests/numerical_analysis/analytical_export/analytical_export 6 0.0 matrices_N6_C0.0.json

# Compute and plot spatio-temporal dispersion with safetyFactor = 0.4
python3 examples/numerical_analysis/full_dispersion/plot_full_dispersion.py matrices_N6_C0.0.json 0.4
```

### 3. Full Automated Sweep
Run the automated parameter sweep across orders $N \in \{2 \dots 10\}$ and fluxes $C_0 \in \{0.0, 0.5, 1.0\}$:
```bash
./examples/numerical_analysis/full_dispersion/run_sweep.sh
```
All generated figures are saved to `examples/numerical_analysis/full_dispersion/results/`.

---

## 5. Visual Output

Each plot `full_dispersion_N{N}_C{C0}.png` contains two panels:
* **Left Panel ($v_p / c$ vs. PPW):** Points Per Wavelength from $628$ down to $2$ on a log scale. Shows full spatio-temporal dispersion (solid circles) alongside pure spatial dispersion (dashed lines) across 4 wave angles $\theta \in [0^\circ, 15^\circ, 30^\circ, 45^\circ]$.
* **Right Panel ($\omega_{\text{num}}$ vs. $|\mathbf{k}|$):** Numerical frequency vs. wavenumber from $0$ to $4.0\text{ rad/m}$, comparing full dispersion against spatial-only dispersion and the pure time (RK45) curve.
