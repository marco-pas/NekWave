# 3D Box PEC Cavity Benchmark (NekCEM Validation Case)

This example directly imports the canonical NekCEM 27-element mesh (`contrib/NekCEM/tests/3dboxpec/3dboxpec.rea`) and executes the 3D electromagnetic cavity resonance test with Perfect Electric Conductor (PEC) boundary conditions.

## Problem Formulation

The 3D Box PEC test simulates an analytical electromagnetic standing wave eigenmode inside a cube $[-1, 1]^3$ enclosed by PEC walls ($\hat{n} \times \mathbf{E} = 0$).

The exact analytical solution is:
$$E_x(x, y, z, t) = -\cos(\pi x) \sin(\pi y) \sin(\pi z) \frac{\sin(\omega t)}{\sqrt{2}}$$
$$E_y(x, y, z, t) =  \sin(\pi x) \cos(\pi y) \sin(\pi z) \frac{\sin(\omega t)}{\sqrt{2}}$$
$$E_z(x, y, z, t) = 0$$

$$H_x(x, y, z, t) = -\sin(\pi x) \cos(\pi y) \cos(\pi z) \frac{\cos(\omega t)}{\sqrt{6}}$$
$$H_y(x, y, z, t) = -\cos(\pi x) \sin(\pi y) \cos(\pi z) \frac{\cos(\omega t)}{\sqrt{6}}$$
$$H_z(x, y, z, t) =  2\cos(\pi x) \cos(\pi y) \sin(\pi z) \frac{\cos(\omega t)}{\sqrt{6}}$$

with eigenfrequency $\omega = \pi \sqrt{3}$.

## Mesh and Boundary Conditions

- **Mesh**: 27 hexahedral elements read directly from `3dboxpec.rea` (`**MESH DATA**` and `**BOUNDARY CONDITIONS**` sections).
- **Boundary**: All boundary faces are tagged `PEC` (or `P  ` in NekCEM `.rea` syntax).
- **DG Order**: $N=4$ ($5^3 = 125$ GLL quadrature points per element, 3,375 total DOFs).
- **Numerical Flux**: Central flux ($c_0 = 0$) for non-dissipative energy conservation.

## Running the Case

```bash
# Run with default parameter file
./build/examples/3dboxpec/3dboxpec examples/3dboxpec/3dboxpec.par

# Run with HDF5 export enabled
./build/examples/3dboxpec/3dboxpec tests/input/3dboxpec.par
```

