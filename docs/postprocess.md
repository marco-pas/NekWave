# Postprocessing and Physical Verification Guide (TODOs)

This document tracks upcoming verification tasks and postprocessing strategies to validate that simulation results are physically meaningful.

---

## 1. Physical Sanity Checks

### A. Total Electromagnetic Energy Conservation
In a closed lossless domain (e.g. enclosed by Perfect Electric Conductor $\hat{n} \times \mathbf{E} = 0$), Poynting's theorem guarantees that the total energy is conserved:

$$U(t) = \frac{1}{2} \int_\Omega \left( \varepsilon |\mathbf{E}|^2 + \mu |\mathbf{H}|^2 \right) d\Omega = \text{const}$$

*   **TODO**: Monitor the relative energy drift:
    $$\Delta U(t) = \frac{|U(t) - U(0)|}{U(0)}$$
    For dissipation-free DG with central fluxes or purely internal pulses before hitting boundaries, $\Delta U \approx 0$ up to time-stepper discretization errors $O(\Delta t^4)$.
    With Upwind fluxes ($C_0 = 1$), slight numerical dissipation across element interfaces is expected and should decrease with polynomial order $p$-refinement.

### B. Involutions (Divergence Constraints)
Maxwell's curl equations implicitly preserve Gauss's laws if satisfied at $t = 0$:

$$\nabla \cdot (\varepsilon \mathbf{E}) = \rho, \quad \nabla \cdot (\mu \mathbf{H}) = 0$$

*   **TODO**: Implement a discrete divergence check using the weak divergence operator:
    $$\|\nabla \cdot \mathbf{B}\|_{L^2} < \text{tolerance}$$

### C. Cavity Resonant Eigenmodes
*   **TODO**: For a rectangular PEC box $[0, L_x] \times [0, L_y] \times [0, L_z]$, initialize with an analytic standing wave mode:
    $$E_x(x, y, z, t) = \dots \cos(\omega t)$$
    Verify that the simulated oscillation frequency $\omega_{\text{sim}}$ matches the exact analytic frequency:
    $$\omega_{mnp} = c \pi \sqrt{\left(\frac{m}{L_x}\right)^2 + \left(\frac{n}{L_y}\right)^2 + \left(\frac{p}{L_z}\right)^2}$$

---

## 2. Data Export & Visualization Strategy

Given visualization challenges with non-standard binary formats, the recommended strategy is:

### Option 1: Lightweight 1D Line Probes (ASCII / CSV)
*   **TODO**: Extract field values along a 1D line (e.g., along the diagonal from $(-1, -1, -1)$ to $(1, 1, 1)$).
*   Save $(s, E_x, E_y, E_z, H_x, H_y, H_z)$ as simple text/CSV files.
*   Can be plotted instantaneously with standard Python (`matplotlib`) or gnuplot without needing ParaView.

### Option 2: Standard VTK Unstructured Grid (`.vtu`)
*   **TODO**: Write an ASCII or binary `.vtu` exporter in `Orchestrator` or a dedicated writer class.
*   Maps each hexahedral element to a `VTK_HEXAHEDRON` or subdivides GLL cells.
*   Directly openable in ParaView or VisIt.

---

## 3. Python Postprocessing Script (`postprocess.py`)

A draft Python script `postprocess.py` will read the log or probed data and plot:
1. $E(t)$ and $H(t)$ field evolution.
2. Energy conservation curve $U(t) / U(0)$.
3. 1D spatial profile snapshots at different times.

