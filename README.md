# NekWave

NekWave is a high-fidelity time-domain Maxwell equation solver designed for direct numerical simulations on modern accelerated high-performance computing architectures.

The solver employs the discontinuous Galerkin spectral element method (DG-SEM) with tensor-product Gauss-Lobatto-Legendre (GLL) quadratures and high-order sum factorization to achieve spectral spatial accuracy alongside explicit low-storage Runge-Kutta time integration.

## Project Structure

```text
NekWave/
├── CMakeLists.txt          # Primary CMake build and CTest configuration
├── README.md               # Project overview and instructions
├── src/                    # C++ source code and headers (.cpp, .hpp)
│   ├── case.hpp / .cpp     # Modular 3-phase simulation lifecycle (preprocess, simulate, postprocess)
│   ├── config.hpp          # Extensible key-value configuration parser and dynamic probes
│   ├── dg_solver.hpp / .cpp # High-level GPU solver coordinator & LSRK45 stepper
│   ├── device/             # Low-level CUDA device kernels and launchers
│   │   ├── dg_kernels.cu   # Raw GPU kernels (volume curl, face flux, mass matrix, LSRK45 update)
│   │   └── dg_kernels.hpp  # C++ launcher dispatch declarations
│   ├── main.cpp            # Application CLI entry point
│   ├── mesh.hpp / .cpp     # Spectral element mesh, GLL nodes, metric terms
│   └── probe.hpp / .cpp    # Observation probes and time-series recording
├── examples/               # Modular Neko-style case suite
│   ├── CMakeLists.txt      # Build and CTest registration for examples
│   ├── cavity_gaussian/    # 3D PEC cavity Gaussian pulse benchmark (.par, .cpp, README)
│   └── cavity_eigenmode/   # Analytical TM_110 standing wave benchmark (.par, .cpp, README)
├── docs/                   # Documentation and technical guides
│   ├── conventions.md      # C++ code and naming conventions
│   ├── method.md           # Mathematical formulation and DG-SEM derivation
│   ├── postprocess.md      # Postprocessing and Fourier mode analysis guide
│   └── progress.md         # Implementation roadmap and status
├── output/                 # Simulation outputs (CSV tables, diagnostic figures)
├── tests/                  # Verification and unit tests
│   └── nekwave_test.cpp    # Unit test suite verifying quadrature, scaling, stability
└── tools/                  # Postprocessing and visualization utilities
    └── postprocess.py      # Multi-panel dashboard and 1D FFT mode extractor
```

## 3-Phase Simulation Lifecycle

Following Neko's architecture (`neko/src/case.f90`), all simulations are organized into three explicit stages:

1. **Preprocessing (`preprocess()`)**:
   - Parses the case `.par` parameter file and spectral element mesh.
   - Generates high-order Gauss-Lobatto-Legendre (GLL) quadrature nodes and computes metric Jacobian transformations.
   - Evaluates initial electromagnetic conditions ($E$, $H$) on host RAM.
   - Registers observation probes and exports initial field distributions (`field_initial.csv`).

2. **Simulation (`simulate()`)**:
   - Allocates GPU memory and uploads mesh metrics and initial fields into GPU VRAM.
   - Advances explicit 5-stage Low-Storage Runge-Kutta 4th-order (LSRK45) time integration entirely on GPU.
   - Evaluates inter-element numerical fluxes and element-local volume curls in device kernels.
   - Streams time-series observations to observation probes and energy logs (`energy_history.csv`).

3. **Postprocessing (`postprocess()`)**:
   - Finalizes probe logs and exports final field states (`field_final.csv`).
   - Releases GPU allocations via `dgSolver_->finalize()`.
   - Executes custom postprocessing hooks (`PostprocessingFn`).

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

CTest automatically runs:
* `MeshQuadratureTests`: Verifies GLL quadrature weights, nodes, symmetry, differentiation matrices, and metric Jacobians.
* `ProbeScalingTests`: Verifies relative-to-physical coordinate transformations for observation probes.
* `MaxwellStabilityTests`: Verifies Maxwell DG-SEM spatial discretization, LSRK45 integration, and discrete energy dissipation monotonicity.
* `NekWaveUnitTests`: Comprehensive test runner.
* `CavityGaussianRun`: Runs the 3D cavity Gaussian pulse example.
* `CavityEigenmodeRun`: Runs the analytical $TM_{110}$ eigenmode example and verifies that discrete $L^2$ error matches spectral precision.

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
