# 3D Box Periodic Benchmark (`3dboxper`)

This example demonstrates electromagnetic wave propagation with periodic boundary conditions on the canonical NekCEM 3D Box Periodic benchmark mesh.

## Overview
- **Mesh**: 128 hexahedral elements ($4 \times 4 \times 8$) loaded directly from `contrib/NekCEM/tests/3dboxper/3dboxper.rea` and its companion binary `3dboxper.re2`.
- **Domain**: $[0, 2\pi]^3$ (rescaled from $[-1, 1]^3$ matching NekCEM's `usrdat2`).
- **Boundary Conditions**: 160 periodic faces across all outer box boundaries ($x, y, z$).
- **Analytical Solution**:
  $$\omega = \sqrt{3}$$
  $$\mathbf{E}(t) = \begin{pmatrix} 0 \\ \cos(x) \sin(y) \sin(z) \cos(\omega t) \\ \cos(x) \cos(y) \cos(z) \cos(\omega t) \end{pmatrix}$$
  $$\mathbf{H}(t) = \frac{\sin(\omega t)}{\omega} \begin{pmatrix} 2 \cos(x) \sin(y) \cos(z) \\ -\sin(x) \cos(y) \cos(z) \\ \sin(x) \sin(y) \sin(z) \end{pmatrix}$$
- **Divergence & Curl**: $\nabla \cdot \mathbf{E} = 0$ and $\nabla \cdot \mathbf{H} = 0$ identically everywhere in the domain.

## Running the Benchmark
```bash
# Run standalone verified benchmark
./build/ctests/3dboxper/3dboxper examples/3dboxper/3dboxper.par

# Or run via main nekwave executable
./build/nekwave examples/3dboxper/3dboxper.par
```

## ParaView Visualization
Snapshots are exported into `output_3dboxper/` as `.vtu` files referenced by `fields.pvd`.
With `export_continuous_vtk = true`, overlapping GLL boundary nodes are merged into a continuous conforming mesh, enabling seamless streamline and iso-contour extraction. Pre-computed derived fields `div_E`, `div_H`, `curl_E`, and `curl_H` are included directly in the output data.
