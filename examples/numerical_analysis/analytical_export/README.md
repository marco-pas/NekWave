# NekWave Reference Element Analytical Matrix Exporter

This directory contains the C++ tool `analytical_export`, which extracts exact reference element Discontinuous Galerkin operators directly from NekWave's core library and exports them to structured JSON.

---

## 1. Overview

Rather than re-implementing spectral element quadrature and differentiation in Python, `analytical_export` hooks directly into NekWave's C++ `Mesh` class. This guarantees that:
* Gauss-Lobatto-Legendre (GLL) quadrature weights $w_i$ and nodes $\xi_i$ match the solver exactly.
* Spectral differentiation matrices $D_{ij}$ match the solver exactly.
* Volumetric divergence and boundary numerical flux formulations ($C_0 = 0.0$ Central, $C_0 = 1.0$ Upwind) are identical to `dg_solver.cpp` and `dg_kernels.cu`.

---

## 2. Mathematical Formulation

The exporter targets the 2D Maxwell Transverse Magnetic (TM) system:

$$\frac{\partial E_z}{\partial t} = \frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y}$$
$$\frac{\partial H_x}{\partial t} = -\frac{\partial E_z}{\partial y}$$
$$\frac{\partial H_y}{\partial t} = \frac{\partial E_z}{\partial x}$$

* **DOFs:** With polynomial order $N$, a 2D element has $N \times N$ GLL nodes. With 3 field components ($E_z, H_x, H_y$), the total degrees of freedom per element is:
  $$\text{dim} = 3 N^2$$
* **Mass Matrix ($\mathbf{M}$):** Block-diagonal matrix of size $3N^2 \times 3N^2$.
* **Stiffness Matrix ($\mathbf{S}$):** Combines internal volumetric spatial differentiation and self-element boundary flux penalties.
* **Interface Flux Matrices ($\mathbf{F}_k$):** For each of the 4 faces ($k \in \{-x, +x, -y, +y\}$), $\mathbf{F}_k$ couples the reference element to its neighbor across physical offset vector $\Delta\mathbf{r}_k = [\Delta x_k, \Delta y_k]$.

---

## 3. JSON Output Structure

The exported JSON file contains:
```json
{
  "order": 4,
  "c0": 0.5,
  "dx": 1.0,
  "dy": 1.0,
  "dxmin": 0.084888,
  "mass_matrix": [ ... ],
  "stiffness_matrix": [ ... ],
  "neighbors": [
    { "face": 0, "offset": [-1.0, 0.0], "flux_matrix": [ ... ] },
    { "face": 1, "offset": [ 1.0, 0.0], "flux_matrix": [ ... ] },
    { "face": 2, "offset": [ 0.0,-1.0], "flux_matrix": [ ... ] },
    { "face": 3, "offset": [ 0.0, 1.0], "flux_matrix": [ ... ] }
  ]
}
```

---

## 4. Compilation & Usage

### Compilation
The executable is built as part of the standard CMake build:
```bash
cmake --build build --target analytical_export -j4
```
The compiled binary is located at `build/ctests/numerical_analysis/analytical_export/analytical_export`.

### Command-Line Arguments
```bash
./build/ctests/numerical_analysis/analytical_export/analytical_export <order> <C0> [output_file.json]
```

* `order` ($N$): Polynomial order ($N \ge 2$, polynomial degree $P = N - 1$).
* `C0`: Numerical flux parameter ($0.0 = \text{Central}, 0.5 = \text{Half-Upwind}, 1.0 = \text{Upwind}$).
* `output_file.json` (optional): Output file path (defaults to `analytical_matrices_order{N}.json`).

### Examples
```bash
# Order N = 4, Central Flux (C0 = 0.0)
./build/ctests/numerical_analysis/analytical_export/analytical_export 4 0.0 matrices_N4_C0.0.json

# Order N = 8, Upwind Flux (C0 = 1.0)
./build/ctests/numerical_analysis/analytical_export/analytical_export 8 1.0 matrices_N8_C1.0.json
```

