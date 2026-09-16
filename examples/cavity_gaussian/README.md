# 3D PEC Cavity Gaussian Pulse Benchmark

## Overview

This case simulates the evolution and reflection of an initial 3D Gaussian electromagnetic pulse in $E_z$ within a closed metallic cavity with Perfect Electric Conductor (PEC) boundary conditions on all six faces.

## Governing Setup

* **Domain**: Cubical cavity $[-1, 1] \times [-1, 1] \times [-1, 1]$ ($L_x = L_y = L_z = 2$).
* **Mesh**: $8 \times 8 \times 8 = 512$ hexahedral elements with polynomial order $N=4$ (125 nodes per element, 64,000 total collocation points).
* **Flux Formulation**: Energy-conserving central flux ($C_0 = 0.0$).
* **Boundary Conditions**: PEC ($\hat{n} \times \mathbf{E} = 0$, $\hat{n} \cdot \mathbf{H} = 0$).
* **Initial Condition**: Localized Gaussian pulse centered at the origin:
  $$E_z(x, y, z, 0) = \exp(-\sigma (x^2 + y^2 + z^2)), \quad \sigma = 20.0$$
  All other field components are initially zero.

## Execution

From the build directory:

```bash
cd build
./examples/cavity_gaussian/cavity_gaussian ../examples/cavity_gaussian/cavity_gaussian.par
```

Or via CTest:

```bash
ctest -R CavityGaussianRun --output-on-failure
```

