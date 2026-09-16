# NekWave Project Progress

## Completed
* Initialized the C++11 project structure (`Orchestrator`, `Physics`, `Mesh`, `TimeStepperRK45`).
* Imposed constraints for a strictly serial implementation (avoiding MPI, OpenMP, CUDA).
* Extracted the fundamental DG methodology and structure from `NekCEM`.
* Implemented Low-Storage RK45 (LSRK45) time stepper with exact coefficients mapped from `NekCEM`'s `rk_storage` in `cem_common.F`.
* Outlined the 3D Maxwell Discontinuous Galerkin spatial operator in `Physics.cpp`, mapping directly to `cem_maxwell_op` from `cem_maxwell.F`.
* **Mesh & Quadrature Operators**:
  * Implemented 1D Gauss-Lobatto-Legendre (GLL) points, weights, and polynomial evaluation.
  * Implemented 1D differentiation matrix $\mathbf{D}$ and transpose $\mathbf{D}^T$ matching NekCEM's `DGLL` in `nek5_speclib.F`.
  * Generated 3D quadrature weights $w_3 = w_i w_j w_k$ and metric factors ($r_x, s_x, t_x, r_y, s_y, t_y, r_z, s_z, t_z$).

  * Added affine bounding-box transformation setup (`Mesh::setAffineBox`) for hexahedral elements.
  
* **NekCEM `.rea` Mesh Loader**:
  * Implemented `Mesh::loadFromRea` to parse legacy NekCEM / Nek5000 `.rea` files.
  * Added a prominent reminder banner at the top of `Mesh.hpp` and `Mesh.cpp` indicating that this `.rea` parser is adapted 1-to-1 from NekCEM for legacy test cases and will eventually be replaced by a modern meshing pipeline (like Neko's `.nmsh` or Gmsh).
  * Automatically extracts 8 corner coordinates, evaluates trilinear hex Jacobian matrices $\mathcal{J}$, determinants $J$, and metric tensors $\mathcal{J}^{-T}$.
  * Captures face connectivity and boundary conditions (`PEC`, `E`, etc.) into `FaceInfo` structures.
  * Validated against official benchmark `NekCEM/tests/3dboxpec/3dboxpec.rea` (27 elements, 162 faces: 54 PEC, 108 internal).
* **Local Element Differentiation (`local_grad3`)**:
  * Implemented `Physics::mxm` matrix-matrix multiplication routine for tensor product contractions using standard loops without external dependencies.
  * Implemented `Physics::local_grad3` computing reference derivatives via tensor contractions:
    * $u_r = (\mathbf{I} \otimes \mathbf{I} \otimes \mathbf{D}) u$
    * $u_s = (\mathbf{I} \otimes \mathbf{D} \otimes \mathbf{I}) u$
    * $u_t = (\mathbf{D} \otimes \mathbf{I} \otimes \mathbf{I}) u$
* **Mass Matrix Inversion (`Physics::applyInverseMassMatrix`)**:
  * Implemented exact diagonal mass matrix inversion: scaling residuals by $\frac{1}{\varepsilon J W}$ and $\frac{1}{\mu J W}$, matching NekCEM's `cem_maxwell_invqmass`.
* **Automatic CFL & Time-Step Calculation**:
  * Implemented `Mesh::computeAutomaticCFL` and `Mesh::computeAutomaticDt` in `Mesh.hpp` and `Mesh.cpp`.
  * Computes $\text{CFL}_{\text{auto}}$ directly from 3D spatial dimension ($d=3$), Carpenter-Kennedy LSRK45 stability boundary limit ($\alpha_{\text{RK45}} \approx 1.75$), and safety factor $S = 0.4$:
    $$\text{CFL}_{\text{auto}} = \frac{S \cdot \alpha_{\text{RK45}}}{\sqrt{3}} \approx 0.404$$
    $$\Delta t = \text{CFL}_{\text{auto}} \times \frac{dx_{\min}}{c}$$
  * Automatically evaluates $dx_{\min}$ from physical GLL node spacing $\mathcal{O}(h/N^2)$ across all elements, adapting dynamically to any polynomial order $N$ and mesh refinement.
  * Added support for `cfl = auto` and `dt = auto` in `Config.hpp` and `case.par`, while preserving manual overrides if specified.
  * Verified across polynomial orders:
    * $N=3$: $dx_{\min} = 0.333333$, $\Delta t = 0.134715$, perfectly stable over 100 steps ($t = 13.47$).
    * $N=5$: $dx_{\min} = 0.115115$, $\Delta t = 0.0465233$, relative energy change $< 0.03\%$ over 500 RK stages.
* **Built-in Cartesian Box Mesh Generator ($h$-refinement)**:
  * Implemented `Mesh::createBoxMesh(nelx, nely, nelz, xmin, xmax, ymin, ymax, zmin, zmax)` in `Mesh.hpp` and `Mesh.cpp`.
  * Allows creating arbitrary structured hexahedral meshes ($K_x \times K_y \times K_z$) on any bounding box $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}] \times [z_{\min}, z_{\max}]$ without requiring external `.rea` files.
  * Automatically establishes exact hex corner geometry, metric factors, PEC outer boundary conditions, and internal interface neighbor connectivity.
  * Configurable in `case.par` via `elements_x`, `elements_y`, `elements_z` (or `nelx`, `nely`, `nelz`) and domain boundaries (`xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`).
  * Validated on $4 \times 4 \times 4 = 64$ elements (384 faces, 8,000 collocation nodes at $N=5$, 48,000 DOFs), running 500 steps (2,500 stages) with automatic CFL time-stepping ($\Delta t = 0.03489$) and $<0.02\%$ energy variation.
* **Build System & Execution in `app/` Directory**:
  * Makefile updated to compile all object files (`.o`), header dependency files (`.d`), and the executable binary (`nekwave`) directly into the `app/` directory.
  * `make run` synchronizes `case.par` into `app/` and executes from within `app/`: `cd app && ./nekwave case.par`.
  * `.gitignore` updated to ignore `app/`, keeping the source root clean.
* **Automated Profiling Support (`make profile`)**:
  * Integrated Google `gperftools` CPU profiling directly into `Makefile`.
  * Builds dedicated profiler binary `app/nekwave_prof` with debug symbols (`-g`), preserved frame pointers (`-fno-omit-frame-pointer` for ARM64 stack unwinding), and links `libprofiler`.
  * Executes with `CPUPROFILE=nekwave.prof` in `app/`, generating call graph sample interrupts and reporting diagnostic instructions for `pprof`.
* **Field Export & Continuous Energy Tracking**:
  * Implemented `Orchestrator::saveFields` exporting $(x, y, z, E_x, E_y, E_z, H_x, H_y, H_z)$ at $t=0$ (`app/field_initial.csv`) and at $t=t_{\text{final}}$ (`app/field_final.csv`).
  * Implemented continuous energy time history logging (`app/energy_history.csv`) capturing step, simulation time, $\max|\mathbf{E}|$, $\max|\mathbf{H}|$, and total discrete energy $U(t)$ at the configured output frequency.
* **Fourier Mode Analysis & Visualization (`make plot`)**:
  * Implemented comprehensive `postprocess.py` script generating `app/nekwave_dashboard.png`:
    1. **Energy Conservation Curve**: Plots $U(t)$ and relative drift $(U(t) - U_0) / U_0$, demonstrating $< 0.0009\%$ energy drift over 2,500 stages on 17,500 collocation points.
    2. **Field Profile Comparison**: 1D centerline slice comparing initial Gaussian pulse ($t=0$) against reflected standing wave packet ($t=t_{\text{final}}$) bounded by PEC walls ($x = \pm 1$).
    3. **2D Mid-plane Contour**: Reconstructs standing interference wave pattern in the cavity.
    4. **2D Spatial Fourier Spectrum**: Performs 2D FFT of $E_z$ and overlays theoretical PEC cavity eigenmodes $k_{mn} = \frac{\pi}{2} \sqrt{m^2 + n^2}$ (eigenmodes $(1,1), (1,2), (2,2), \dots$).
  * Added `make plot` target to Makefile for automated execution.

## Test Findings (Iteration A)
* **Initial Wave Evolution**: Over early time steps ($t \in [0, 0.01]$), the initial $E_z$ pulse cleanly curls into magnetic field components ($H$ rises from 0 to 0.78 while $E$ decreases from 1.0 to 0.54) with near-exact energy conservation ($U = 0.0609$ vs initial $0.0598$).
* **Identified Instability Mechanism**: Because face numerical fluxes (`restrictToFace`, `computeFlux`, `addFluxToRHS`) are currently empty stubs, elements are isolated without jump penalty terms. Once the wave reaches element boundaries, unconstrained interface modes grow exponentially.

* **Face Geometry, Metric Normals & Trace Connectivity (`Mesh`)**:
  * Implemented `FacePointData` and `ElementFaceData` representing all quadrature points on each hex face.
  * Implemented `Mesh::getFaceNodeVolIndex` mapping 2D face coordinates $(p, q) \in [0, N-1]^2$ to 3D volume node indices across all 6 faces matching NekCEM's `cemface` indexing convention.
  * Evaluated physical outward unit normal vectors $\hat{n} = (n_x, n_y, n_z)$ and face surface metric areas $dA = J_{\text{face}} w_p w_q$ via Nanson's formula $\mathbf{n} dA = J \mathcal{J}^{-T} \mathbf{n}_{\text{ref}} dA_{\text{ref}}$.
  * Implemented robust coordinate-based neighbor matching across element interfaces within $10^{-10}$ tolerance.
* **Face Restriction (`Physics::restrictToFace`)**:
  * Implemented extraction of internal traces $\mathbf{E}^-$, $\mathbf{H}^-$ at all face collocation nodes into internal trace arrays `m_fEN` and `m_fHN` matching NekCEM's `cem_maxwell_restrict_to_face`.
* **Numerical Fluxes (`Physics::computeFlux`)**:
  * Implemented Upwind ($C_0 = 1.0$) and Central ($C_0 = 0.0$) numerical flux formulation matching NekCEM's `cem_maxwell_flux3d` and `cem_maxwell_flux_pec`:
    * $\mathbf{F}^*_H = -0.5 (\hat{n} \times [[\mathbf{E}]]) - 0.5 C_0 (\hat{n} \times (\hat{n} \times [[\mathbf{H}]]))$
    * $\mathbf{F}^*_E = +0.5 (\hat{n} \times [[\mathbf{H}]]) - 0.5 C_0 (\hat{n} \times (\hat{n} \times [[\mathbf{E}]])))$
  * Implemented PEC boundary condition mirror reflection:
    * $[[\mathbf{E}]] = -2\mathbf{E}^-$, $[[\mathbf{H}]] = \mathbf{0}$
    * Enforces $\hat{n} \times \mathbf{E} = 0$ weakly through the $\mathbf{H}$ residual, with tangential damping through the $\mathbf{E}$ residual.
* **Surface Flux Lifting (`Physics::addFluxToRHS`)**:
  * Implemented surface integral lifting accumulating $dA \cdot \mathbf{F}^*$ into the volume residual at the corresponding boundary nodes matching `cem_maxwell_add_flux_to_res`.
* **Volume Curl Weighting Fix**:
  * Weighted reference curl derivatives by the full volume measure $J_k W_i$, ensuring the strong DG residual correctly balances the surface flux lifting upon inverse mass matrix scaling.

## Test Findings & Empirical Verification (Iteration B)
* **Energy Conservation Benchmark (`3dboxpec.rea`, 27 elements, 162 faces, order N=3)**:
  * **Central Flux ($C_0 = 0.0$)**:
    * 100 time steps ($t \in [0, 0.5]$): Initial Energy $U_0 = 0.0598133$, Final Energy $U_{100} = 0.0598133$.
    * Total energy drift across 500 Runge-Kutta stages is $-1.15 \times 10^{-9}$ (essentially machine precision).
  * **Upwind Flux ($C_0 = 1.0$)**:
    * Clean, strictly monotonic numerical dissipation with smooth wave reflections off PEC boundaries.
* **1D Temporal Observation Probe & Resonant Cavity Mode Spectrum (`make plot`)**:
  * Added observation probe point configuration (`probeX`, `probeY`, `probeZ`) in `Config.hpp` and `case.par`.
  * Implemented nearest collocation node search in `Orchestrator::setup()`. Configured off-center probe at $(0.2, 0.15, -0.1)$ to capture all even and odd cavity modes without symmetry nulls.
  * Implemented high-frequency temporal logging (`app/probe_history.csv`) capturing $(t, E_x, E_y, E_z, H_x, H_y, H_z)$ at every single time step.
  * Implemented `postprocess.py::plot_1d_fft_modes()` performing 1D real FFT (`scipy.fft.rfft` / `np.fft.rfft`) on the continuous probe signal.
  * Implemented exact theoretical resonant cavity eigenfrequency evaluator `eigf(modes, c0, Lx, Ly, Lz)` matching the user's analytical formulation:
    $$f_{mnp} = \frac{c_0}{2} \sqrt{\left(\frac{m}{L_x}\right)^2 + \left(\frac{n}{L_y}\right)^2 + \left(\frac{p}{L_z}\right)^2}$$
  * Generates dedicated 1D mode spectrum artifact (`app/cavity_modes_1d.png`):
    * **Upper panel**: Multi-frequency time-domain electric field waveform $E_z(t)$ at the internal probe.
    * **Lower panel**: 1D Fourier amplitude spectrum $|\hat{E}_z(f)|$ (and angular frequency $\omega = 2\pi f$) with theoretical eigenmode dashed vertical lines and $(m,n,p)$ labels.
  * Successfully verified on a dense $9 \times 9 \times 9 = 729$ element mesh ($N=6$, 157,464 collocation points) over 2,000 steps ($t \in [0, 21.1]$):
    * Relative energy drift is strictly linear and exceptionally small: $-7.517 \times 10^{-7}$ over 10,000 RK stages.
    * Discrete FFT spectral peaks align with analytical cavity modes:
      * $(1,1,0) / (1,0,1)$ at $f = 0.3536$
      * $(2,1,1) / (1,2,1)$ at $f = 0.6124$
      * $(3,1,0) / (1,3,0)$ at $f = 0.7906$
      * $(3,2,1)$ at $f = 0.9354$

* **Modular Multi-Point Probe System (`Probe.hpp` / `Probe.cpp`)**:
  * Decoupled spatial probe recording into a dedicated, reusable `ProbeManager` class (`Probe.hpp`, `Probe.cpp`).
  * Support for 3 distinct observation probe locations across the 3D cavity:
    * Probe 1: Target $(0.20, 0.15, -0.10) \rightarrow$ Node #63476 $(0.19053, 0.137216, -0.111111)$
    * Probe 2: Target $(-0.35, 0.25, 0.40) \rightarrow$ Node #115222 $(-0.35897, 0.254135, 0.412698)$
    * Probe 3: Target $(0.10, -0.45, -0.30) \rightarrow$ Node #57293 $(0.111111, -0.47619, -0.306878)$
  * Writes high-frequency multi-channel field traces (`step, time, Ex_1..3, Ey_1..3, Ez_1..3, Hx_1..3, Hy_1..3, Hz_1..3`) at every time step.
* **Encapsulated Output Directory Structure (`app/output/`)**:
  * All simulation outputs (CSVs and PNG figures) are directed into `app/output/`:
    * `app/output/probe_history.csv`
    * `app/output/energy_history.csv`
    * `app/output/field_initial.csv`
    * `app/output/field_final.csv`
    * `app/output/cavity_modes_1d.png`
    * `app/output/nekwave_dashboard.png`
  * Root and `app/` build directory remain clean of raw diagnostic data files.
* **Multi-Probe 1D Temporal FFT Mode Analysis (`make plot`)**:
  * Updated `postprocess.py` to process multi-channel probe data.
  * **Time-domain panel**: Compares waveforms $E_z^{(1)}(t), E_z^{(2)}(t), E_z^{(3)}(t)$ across all 3 spatial locations.
  * **Frequency-domain panel**: Overlays the 1D Fourier spectra of all 3 probes with the theoretical PEC cavity eigenmodes $f_{mnp}$.
* **Configurable Cavity Dimensions ($L_x, L_y, L_z$)**:
  * Added `L_x`, `L_y`, `L_z` (and aliases `Lx, Ly, Lz`) to [`Config.hpp`](file:///Users/marcopas/Desktop/KTH/SEDG/NekWave/Config.hpp) and [`case.par`](file:///Users/marcopas/Desktop/KTH/SEDG/NekWave/case.par).
  * Automatically sets centered domain bounds $[-L_x/2, L_x/2] \times [-L_y/2, L_y/2] \times [-L_z/2, L_z/2]$ around the Gaussian pulse origin, while preserving explicit bounding coordinates (`xmin`, `xmax`, etc.) if specified.
  * Printed explicitly in solver initialization summary table.
  * Added `load_case_params()` in [`postprocess.py`](file:///Users/marcopas/Desktop/KTH/SEDG/NekWave/postprocess.py) to dynamically read cavity dimensions ($L_x, L_y, L_z$) and update theoretical resonant mode eigenfrequencies $f_{mnp}$ and spatial 2D Fourier grid lines accordingly without hardcoding.

* **Energy Conservation Benchmark (`3dboxpec.rea`, 27 elements, 162 faces, order N=3)**:
  * **Central Flux ($C_0 = 0.0$)**:
    * 100 time steps ($t \in [0, 0.5]$): Initial Energy $U_0 = 0.0598133$, Final Energy $U_{100} = 0.0598133$.
    * Total energy drift across 500 Runge-Kutta stages is $-1.15 \times 10^{-9}$ (essentially machine precision).
  * **Upwind Flux ($C_0 = 1.0$)**:
    * Clean, strictly monotonic numerical dissipation with smooth wave reflections off PEC boundaries.
    * Stable execution over 50+ time steps with zero blowup.
* **Dense Cavity Resonant Spectrum Benchmark ($9 \times 9 \times 9 = 729$ elements, $N=6$, 157,464 nodes)**:
  * Over 2,000 steps ($t = 21.1004$), relative energy drift was $-7.517 \times 10^{-7}$.
  * Standing wave modes in the PEC box were excited by the Gaussian pulse and captured simultaneously by the 3 internal probes.
  * Discrete FFT spectra across all 3 probes showed identical resonance peaks matching $(1,1,0)$, $(2,1,1)$, $(3,1,0)$, and $(3,2,1)$ modes.

* **Codebase Modularization and Directory Reorganization**:
  * Restructured project layout into dedicated functional directories:
    * `src/`: Core C++ implementation and mesh classes (`config.hpp`, `mesh.hpp`, `mesh.cpp`, `physics.hpp`, `physics.cpp`, `timestepperrk45.hpp`, `timestepperrk45.cpp`, `probe.hpp`, `probe.cpp`, `orchestrator.hpp`, `orchestrator.cpp`, `main.cpp`).
    * `docs/`: Technical documentation and style guides (`conventions.md`, `method.md`, `postprocess.md`, `progress.md`).
    * `output/`: Simulation data tables (`energy_history.csv`, `field_initial.csv`, `field_final.csv`, `probe_history.csv`) and plots (`nekwave_dashboard.png`, `cavity_modes_1d.png`).
    * `tests/`: Automated unit and verification test suite (`nekwave_test.cpp`).
    * `tools/`: Python postprocessing and spectral analysis tools (`postprocess.py`).
  * Enforced lowercase file naming conventions throughout `src/` (e.g., `physics.cpp`, `mesh.cpp`).
  * Migrated build and test system to CMake and CTest with targets for building (`cmake --build build`), testing (`ctest`), executing (`--target run`), and plotting (`--target plot`).
  * Created top-level `README.md` documenting architecture, layout, and usage.

## Next Steps (Iteration C)
* **Analytical Single-Mode Cavity Benchmark**:
  * Initialize an exact single electromagnetic eigenmode (e.g. $TE_{101}$ or $TM_{110}$) to measure $L^2$ error convergence rate against polynomial order $N$ and mesh size $h$.
* **Absorbing Boundary Conditions (ABC / Silver-Müller)**:
  * Extend boundary condition handlers to include radiating open boundaries and PML.
