# NekWave Numerical Analysis Suite

This directory contains the tools and studies for analyzing the numerical properties (stability and dispersion) of the NekWave DGTD solver.

## Analytical Dispersion Framework (Bloch-Floquet Theory)

Instead of simulating a large, multi-element mesh, the analytical dispersion analysis isolates **a single reference element** and applies the Bloch-Floquet plane wave ansatz across its faces.

```
                  Top Neighbor: U_top = U_0 * exp(-j * k_y * dy)
                                    ▲
                                    │ Face 4 (+y)
                      ┌─────────────┴─────────────┐
                      │  •     •     •     •     •│
                      │                           │
   Left Neighbor:     │  •     •     •     •     •│     Right Neighbor:
U_left = U_0 *        │      N x N GLL Nodes      │  U_right = U_0 *
exp(+j * k_x * dx) ◄──┤  •     •     •     •     •├──► exp(-j * k_x * dx)
  Face 1 (-x)         │                           │    Face 2 (+x)
                      │  •     •     •     •     •│
                      │                           │
                      │  •     •     •     •     •│
                      └─────────────┬─────────────┘
                                    │ Face 3 (-y)
                                    ▼
                Bottom Neighbor: U_bottom = U_0 * exp(+j * k_y * dy)
```

### How the Single Element Yields Analytical Modes:
1. **Semi-Discrete Equation**:
   $$\mathbf{M} \frac{d\mathbf{U}_0}{dt} = \mathbf{S} \mathbf{U}_0 + \sum_{k \in \{\text{faces}\}} \mathbf{F}_k \mathbf{U}_k$$
2. **Bloch-Floquet Periodic Boundary Phase Shift**:
   $$\mathbf{U}_k = \mathbf{U}_0 \, e^{-j (\mathbf{k} \cdot \Delta\mathbf{r}_k)}$$
3. **Local Closed Eigenvalue Problem**:
   Substituting the phase shifts into the flux terms decouples the element from an infinite grid:
   $$\frac{d\mathbf{U}_0}{dt} = \mathbf{H}(\mathbf{k}) \mathbf{U}_0, \qquad \mathbf{H}(\mathbf{k}) = \mathbf{M}^{-1} \left[ \mathbf{S} + \sum_k \mathbf{F}_k \, e^{-j (\mathbf{k} \cdot \Delta\mathbf{r}_k)} \right]$$
4. **Eigenvalues $\lambda(\mathbf{k})$**:
   The eigenvalues of $\mathbf{H}(\mathbf{k})$ directly give the numerical frequency $\omega_{\text{num}}(\mathbf{k})$ and damping $\sigma(\mathbf{k})$ without running any time-stepping simulation!

---

### Physical Interpretation of $k$, $\omega$, and Real-World Dimensionalization

#### 1. What $k$ and $\omega_{\text{num}}$ Represent
* **$\mathbf{k}$ (Wavevector) is set by us:**
  It defines the spatial monochromatic plane wave $e^{j(\mathbf{k} \cdot \mathbf{x} - \omega t)}$ we want to test on the mesh. Its wavelength is $\lambda = \frac{2\pi}{|\mathbf{k}|}$, and the grid resolution is given by Points Per Wavelength:
  $$\text{PPW} = \frac{\lambda}{\Delta x} = \frac{2\pi}{|\mathbf{k}| \, \Delta x}$$
* **$\omega_{\text{num}}$ is the numerical scheme's response:**
  Nature dictates that in continuous vacuum, a wave of wavenumber $k$ propagates at $\omega_{\text{exact}} = c |\mathbf{k}|$. However, when discretized with DGTD and Runge-Kutta, the grid responds at an effective numerical frequency $\omega_{\text{num}}(\mathbf{k})$.
  * $v_p / c = \omega_{\text{num}} / (c |\mathbf{k}|) = 1.0$: Exact, zero dispersion error.
  * $v_p / c < 1.0$: Numerical phase lag (short waves propagate too slowly).
  * $v_p / c > 1.0$: Numerical phase lead (waves propagate too quickly).

#### 2. Relating Dimensionless Code Values to Real-World Physics
NekWave non-dimensionalizes parameters ($c_{\text{code}} = 1.0$, $\Delta x_{\text{code}} = 1.0$). To relate code outputs to a real-world simulation with physical element size $\Delta x_{\text{phys}}$ (in meters) and speed of light $c_0 \approx 3 \times 10^8\text{ m/s}$:

The simulation time scale is $[T] = \frac{\Delta x_{\text{phys}}}{c_0}$, leading to the conversion for physical frequency in Hertz:
$$f_{\text{phys}} = \frac{\omega_{\text{code}}}{2\pi} \cdot \left(\frac{c_0}{\Delta x_{\text{phys}}}\right)$$

For example, when **$\omega_{\text{code}} = 1.0$** (where one wavelength spans $\lambda \approx 6.28 \cdot \Delta x_{\text{phys}}$):
| Physical Element Size ($\Delta x_{\text{phys}}$) | Real Physical Frequency ($f_{\text{phys}}$) | Application / Spectrum Band |
| :--- | :--- | :--- |
| **$1.0\text{ m}$** (large structures / aircraft) | **$47.7\text{ MHz}$** | VHF Radio Broadcast |
| **$10\text{ cm} = 0.1\text{ m}$** (RF / accelerator cavities) | **$477\text{ MHz}$** | UHF / Particle Accelerator Cavities |
| **$1\text{ cm} = 0.01\text{ m}$** (circuit boards / antennas) | **$4.77\text{ GHz}$** | 5 GHz Wi-Fi / C-Band Satellite |
| **$1\text{ mm} = 10^{-3}\text{ m}$** (chips / packaging) | **$47.7\text{ GHz}$** | 5G Millimeter-Wave / Radar |
| **$1\ \mu\text{m} = 10^{-6}\text{ m}$** (integrated photonics) | **$47.7\text{ THz}$** | Mid-Infrared Optics |

Because the relative error $(v_p/c - 1)$ is completely dimensionless, the dispersion curves apply identically across all these physical scales!

#### 3. Safety Factor: Stability (Magnitude) vs. Dispersion (Phase)

In `run_sweep.sh`, the time step is computed automatically via the `SAFETY` parameter:
$$\text{CFL}_{\text{auto}} = \text{safetyFactor} \cdot \left(\frac{\alpha_{\text{RK}}}{\sqrt{3}}\right), \qquad \Delta t = \text{CFL}_{\text{auto}} \cdot \frac{\Delta x_{\min}}{c}$$
* `SAFETY = 0.4` (Default): Uses **40%** of the theoretical maximum stable time step, providing safety against grid deformations and boundary reflections.
* `SAFETY = 1.0`: Operates at **100% of the theoretical stability limit** (no safety margin).
* `SAFETY = 0.0`: Means $\Delta t = 0$ (the clock would not advance).

##### The Crucial Distinction: Magnitude vs. Phase
When the time-stepper advances the solution by one time step $\Delta t$, the mode is multiplied by a complex amplification factor:
$$G(z) = |G(z)| \, e^{j \phi}$$

* **Magnitude $|G(z)| \longleftrightarrow$ Stability & Dissipation:**
  * $|G(z)| \le 1$: The scheme is **stable** (energy is bounded).
  * $|G(z)| = 1$: Strictly energy-conserving (neutral stability, zero dissipation).
  * $|G(z)| > 1$: The scheme is **unconditionally unstable**! The solution will blow up exponentially ($U^n \propto |G|^n \to \infty$).
* **Phase $\phi = \arg(G(z)) \longleftrightarrow$ Dispersion:**
  * The numerical frequency is $\omega_{\text{num}} = \frac{\phi}{\Delta t}$.
  * Dictates whether the wave travels at the correct physical speed ($v_p = \omega_{\text{num}} / k \approx c$).

> **Important Takeaway:** A numerical scheme can be **completely non-dispersive** (perfect phase speed $v_p = c$), yet be **catastrophically unstable** if $|G(z)| = 1.001$! Conversely, a scheme can be unconditionally stable (heavily damped) yet suffer from unacceptable dispersion errors.
> * The **Neumann stability analysis** verifies the **magnitude** ($|G(z)| \le 1$).
> * The **Bloch-Floquet dispersion analysis** verifies the **phase** ($v_p / c \approx 1.0$).
> Both must be verified to guarantee a physically faithful simulation.

---

## 1. Neumann Stability ([`neumann_stability/`](neumann_stability/README.md))
Analyzes the stability of the numerical scheme by computing the exact spatial eigenvalues of the reference element DGTD operators ($\mathbf{M}$, $\mathbf{S}$, $\mathbf{F}$) and overlaying their scaled values ($z = \lambda \Delta t$) on the stability region of the Low-Storage RK45 time-integration scheme ($|G(z)| \le 1$). Used to determine maximum stable CFL numbers for different numerical fluxes (Central $C_0 = 0$, Upwind $C_0 = 1.0$) across polynomial orders $N \in \{2 \dots 10\}$.

## 2. Full Analytical Dispersion ([`full_dispersion/`](full_dispersion/README.md))
Computes the exact theoretical spatio-temporal dispersion relation ($v_p / c$ vs. PPW and $\omega$ vs. $k$). Combines the spatial Bloch-Floquet amplification matrix $\mathbf{H}(\mathbf{k})$ with the Carpenter & Kennedy (1994) LSRK45 stability polynomial $G(z)$ to extract the coupled discrete numerical frequency $\omega_{\text{num}} = \arg(G(z)) / \Delta t$ and phase velocity $v_p = \omega_{\text{num}} / k$. Tracks the physical propagating eigenmode across $k \in [0.01, \pi / \Delta x]$ for propagation angles $\theta \in [0^\circ, 15^\circ, 30^\circ, 45^\circ]$.

## 3. Numerical Dispersion ([`numerical_dispersion/`](numerical_dispersion/README.md))
Validates the theoretical analytical dispersion by running actual 3D CUDA GPU simulations on an $L_x = 16\pi$ periodic domain with 64 elements. Uses the **Multi-Harmonic Discrete Modal Phase Velocity** method:
* Simultaneously excites 5 discrete plane wave modes spaced linearly from $k = 0.25$ to $3.75\text{ rad/m}$ ($n = [2, 9, 16, 23, 30]$ integer cycles across $L_x$), guaranteeing strict $C^\infty$ periodic continuity without jump discontinuities.
* Records electromagnetic field time series at two collinear observation probes separated by $d = 1.0$.
* Applies 4-term Blackman-Harris windowing ($-92\text{ dB}$ sidelobe suppression) and 65,536-bin zero-padded FFT with parabolic sub-bin interpolation to isolate the numerical frequency $\omega_{\text{num}}(m)$ of each mode.
* Computes the numerical phase velocity $v_p(m) = \omega_{\text{num}}(m) / k_{\text{true}}(m)$ and compares it against exact physical propagation.

## Common Tools
- [`analytical_export/`](analytical_export/README.md): A C++ utility that directly hooks into the NekWave core `Mesh` class to export the exact reference element mass ($\mathbf{M}$), stiffness ($\mathbf{S}$), and interface flux ($\mathbf{F}$) matrices to JSON for the Python analytical solvers.
- [`tools/analytical_spatial_dispersion.py`](../../tools/analytical_spatial_dispersion.py): Shared Python module for parsing exported JSON matrices, constructing the spatial amplification matrix $\mathbf{H}(\mathbf{k})$, and computing Bloch-Floquet eigenvalues.
