# Analytical Cavity TM_110 Standing Wave Eigenmode Benchmark

## Overview

This benchmark simulates an exact analytical standing wave eigenmode ($TM_{110}$) within a three-dimensional rectangular metallic cavity with PEC boundaries.

## Analytical Solution

For a cavity of dimensions $L_x \times L_y \times L_z$ with coordinates shifted to $\tilde{x} = x - x_{\min}$, $\tilde{y} = y - y_{\min}$, the exact field equations are:

$$E_z(x, y, z, t) = E_0 \sin\left(\frac{m \pi \tilde{x}}{L_x}\right) \sin\left(\frac{n \pi \tilde{y}}{L_y}\right) \cos(\omega t)$$

$$H_x(x, y, z, t) = -\frac{E_0 k_y}{\mu \omega} \sin\left(\frac{m \pi \tilde{x}}{L_x}\right) \cos\left(\frac{n \pi \tilde{y}}{L_y}\right) \sin(\omega t)$$

$$H_y(x, y, z, t) = \frac{E_0 k_x}{\mu \omega} \cos\left(\frac{m \pi \tilde{x}}{L_x}\right) \sin\left(\frac{n \pi \tilde{y}}{L_y}\right) \sin(\omega t)$$

$$E_x = E_y = H_z = 0$$

where:
* $k_x = \frac{m \pi}{L_x}, \quad k_y = \frac{n \pi}{L_y}$
* $\omega = c \sqrt{k_x^2 + k_y^2}$

## Verification

The case driver implements a custom postprocessing hook that compares the numerical field $E_z^{\text{num}}$ with the closed-form analytical solution $E_z^{\text{exact}}$ at $t = t_{\text{final}}$, computing:

$$\|E_z^{\text{num}} - E_z^{\text{exact}}\|_{L^2} = \sqrt{\frac{\sum_k J_k w_k (E_z^{\text{num}} - E_z^{\text{exact}})^2}{\sum_k J_k w_k}}$$

$$\|E_z^{\text{num}} - E_z^{\text{exact}}\|_{L^\infty} = \max_k |E_z^{\text{num}} - E_z^{\text{exact}}|$$

## Execution

From the build directory:

```bash
cd build
./examples/cavity_eigenmode/cavity_eigenmode ../examples/cavity_eigenmode/cavity_eigenmode.par
```

Or via CTest:

```bash
ctest -R CavityEigenmodeRun --output-on-failure
```

