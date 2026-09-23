# NekWave

NekWave is a high-fidelity time-domain Maxwell equation solver designed for direct numerical simulations on modern accelerated high-performance computing architectures.

The solver employs the discontinuous Galerkin spectral element method (DG-SEM) with tensor-product Gauss-Lobatto-Legendre (GLL) quadratures and high-order sum factorization to achieve spectral spatial accuracy alongside explicit low-storage Runge-Kutta time integration.

## 3-Phase Simulation Lifecycle

Following Neko's architecture (`neko/src/case.f90`), all simulations are organized into three explicit stages:

1. **Preprocessing (`preprocess()`)**:

2. **Simulation (`simulate()`)**:

3. **Postprocessing (`postprocess()`)**:

## Getting Started

### Prerequisites

* CMake 3.18 or higher
* Modern C++ compiler supporting C++11 or later (`g++` or `clang++`)
* NVIDIA GPU and CUDA Toolkit (`nvcc`) (Mandatory)
* Python 3 with `numpy`, `matplotlib`, and `scipy`
* Optional: `gperftools` for profiling

### Building with CMake

Configure and build with CMake (CUDA is required):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

To specify a particular compiler (e.g. Apple Clang):

```bash
cmake -B build -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Running Verification and Example Tests with CTest

NekWave uses CTest to orchestrate both low-level unit tests and modular example regression runs:

```bash
ctest --test-dir build --output-on-failure
```

### Running Examples Individually

Each example case in `examples/` compiles into its own dedicated executable:

```bash
cd build

# 3D PEC Cavity Gaussian Pulse
./examples/cavity_gaussian/cavity_gaussian ../examples/cavity_gaussian/cavity_gaussian.par

# Analytical TM_110 Eigenmode Benchmark
./examples/cavity_eigenmode/cavity_eigenmode ../examples/cavity_eigenmode/cavity_eigenmode.par
```

### Running the General Application

To run a simulation using the default example case via CMake:

```bash
cmake --build build --target run
```

Or pass any case file directly to the `nekwave` binary:

```bash
./build/nekwave examples/cavity_gaussian/cavity_gaussian.par
./build/nekwave examples/cavity_eigenmode/cavity_eigenmode.par
```

### Postprocessing and Visualization

Generate the diagnostic dashboard and resonant mode Fourier spectra:

```bash
cmake --build build --target plot
```

Generated figure:
* `output/nekwave_dashboard.png`: Unified analysis dashboard with relative energy drift, 1D centerline field cut ($t=0$ vs $t=t_{\text{final}}$), and full-width 1D multi-probe Fourier spectra with theoretical cavity mode markers.
