# NekWave Neumann Stability Analysis

This tool evaluates the numerical stability of the NekWave DGTD solver by overlaying the continuous spatial eigenspectrum of the reference element DGTD operators directly onto the stability region of the 5-stage Low-Storage Runge-Kutta (LSRK45) time-integration scheme.

---

## 1. Theoretical Background

For the semi-discrete linear system:

$$\frac{d\mathbf{U}_0}{dt} = \mathbf{H}(\mathbf{k}) \mathbf{U}_0$$

advancing by time step $\Delta t$ with an explicit Runge-Kutta method multiplies each spatial eigenmode by the temporal amplification factor $G(z)$, where $z = \lambda \Delta t$ and $\lambda \in \operatorname{spec}(\mathbf{H}(\mathbf{k}))$.

### The von Neumann Stability Criterion
The numerical scheme is **stable** if and only if all scaled spatial eigenvalues lie within the temporal stability domain $\mathcal{S}$:

$$\mathcal{S} = \big\{ z \in \mathbb{C} \;:\; |G(z)| \le 1 \big\}$$

* **$|G(z)| < 1$:** The mode is dissipated (amplitude decays monotonically with time).
* **$|G(z)| = 1$:** Neutral stability (energy-conserving, zero numerical dissipation).
* **$|G(z)| > 1$:** **Unconditionally unstable!** The mode grows exponentially ($U^n \propto |G|^n \to \infty$), causing the simulation to blow up.

### Flux Formulation Effects ($C_0$)
* **Central Flux ($C_0 = 0.0$):** Energy-conserving. The spatial eigenvalues $\lambda$ are strictly imaginary ($\operatorname{Re}(\lambda) = 0$). Stability requires that the scaled imaginary span fits entirely within the imaginary axis stability interval of LSRK45:
  $$[-j \beta_{\text{imag}}, +j \beta_{\text{imag}}] \subset \mathcal{S}$$
* **Upwind Flux ($C_0 > 0.0$):** Dissipative. High-frequency spatial eigenvalues bend into the stable left half-plane ($\operatorname{Re}(\lambda) < 0$), providing numerical dissipation that suppresses spurious grid-scale oscillations.

---

## 2. Automatic Time Step & CFL Calculation

The scaling $z = \lambda \Delta t$ depends on the time step $\Delta t$, which is computed to match NekWave's core solver:

$$\text{CFL}_{\text{auto}} = \text{safetyFactor} \cdot \left(\frac{\alpha_{\text{RK}}}{\sqrt{3}}\right), \qquad \Delta t = \frac{\text{CFL}_{\text{auto}} \Delta x_{\min}}{c}$$

* $\alpha_{\text{RK}} \approx 1.75$: 1D stability limit for Carpenter & Kennedy LSRK45.
* $\sqrt{3}$: Dimensional reduction factor for 3D stability.
* $\Delta x_{\min}$: Smallest spacing between adjacent Gauss-Lobatto-Legendre (GLL) nodes in the element.
* `safetyFactor`: Fraction of theoretical stability limit (default $0.4$, representing $40\%$ of the maximum stable time step).

---

## 3. Pipeline & Usage

### 1. Compile Matrix Exporter
```bash
cmake --build build -j4
```

### 2. Single Run (Export & Plot)
```bash
# Export reference element matrices for Order N = 6, Flux C0 = 0.5
./build/ctests/numerical_analysis/analytical_export/analytical_export 6 0.5 matrices_N6_C0.5.json

# Plot stability region and overlay spatial eigenvalues (with safetyFactor = 0.4)
python3 examples/numerical_analysis/neumann_stability/plot_neumann.py matrices_N6_C0.5.json 0.4
```

### 3. Full Automated Sweep
Run the automated parameter sweep across polynomial orders $N \in \{2 \dots 10\}$ and fluxes $C_0 \in \{0.0, 0.5, 1.0\}$:
```bash
./examples/numerical_analysis/neumann_stability/run_sweep.sh
```
All generated figures are saved to `examples/numerical_analysis/neumann_stability/results/`.

---

## 4. Visual Output & Interpretation

Each generated figure `neumann_N{N}_C{C0}.png` shows:
* **Green Region:** The stable domain $\mathcal{S}$ where $|G(z)| \le 1$.
* **Black Contour:** The boundary of marginal stability $|G(z)| = 1.0$.
* **Red Points:** Scaled spatial eigenvalues $z = \lambda \Delta t$ sampled across wave angles $\theta \in [0^\circ, 15^\circ, 30^\circ, 45^\circ]$ and wavenumbers $k \in [0.01, \pi / \Delta x]$.

**Interpretation:**
* If all red points lie inside the green shaded area, the simulation is guaranteed stable for the given CFL and flux parameter.
* If red points cross outside the boundary into the white domain, the CFL number exceeds the stability limit and must be reduced.

