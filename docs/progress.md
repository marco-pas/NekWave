# NekWave Project Progress

## 1. Completed Milestones

### Core Architecture & C++ Lifecycle
* **Modular 3-Phase Simulation Lifecycle (`Case` Class)**:
  * Implemented `Case` lifecycle (`preprocess()`, `simulate()`, `postprocess()`) conforming to high-order CFD/CEM architectures.
  * Decoupled configuration loading (`Config`), domain generation (`Mesh`), GPU solver execution (`DgSolver`), observation diagnostics (`ProbeManager`), and output streaming.
* **Mesh & Quadrature Operators**:
  * 1D Gauss-Lobatto-Legendre (GLL) points, weights, and high-order Lagrange polynomial evaluation.
  * 1D differentiation matrix $\mathbf{D}$ and transpose $\mathbf{D}^T$ matching NekCEM's `DGLL` in `nek5_speclib.F`.
  * Generated 3D quadrature weights $w_3 = w_i w_j w_k$ and metric factors ($r_x, s_x, t_x, r_y, s_y, t_y, r_z, s_z, t_z$).
  * Built-in Cartesian multi-element box mesh generator (`createBoxMesh`) supporting arbitrary $h$-refinement and periodic boundary conditions across all three Cartesian axes.
  * Preserved backward compatibility for NekCEM `.rea` mesh parsing.

### GPU-Resident CUDA Solver Engine (`src/device/`, `src/dg_solver.cpp`)
* **Fully GPU-Resident Maxwell DGTD Solver**:
  * All state vectors (`d_state_`), Runge-Kutta registers (`d_k_`), residuals (`d_rhs_`), and geometric metrics are allocated directly in GPU VRAM. Zero host-device transfers occur during time stepping.
* **Fused 3D Tensor Contraction Volume Curl Kernel (`gpu_volume_curl_kernel`)**:
  * Evaluates the discrete curl operator for 3D Maxwell fields ($+\nabla \times \mathbf{H}$ for $\mathbf{E}$, $-\nabla \times \mathbf{E}$ for $\mathbf{H}$).
  * Operates with 1 thread block per hex element and $N^3$ threads per block.
  * Uses fast on-chip dynamic shared memory to cache 1D derivative matrix $\mathbf{D}$ ($N^2$ doubles) and vector field components ($3 N^3$ doubles), performing tensor product contractions (`local_grad3`) with unrolled register loops.
* **Surface Flux & Trace Kernels**:
  * `gpu_restrict_faces_kernel`: Restricts interior volume fields to boundary and interface element faces.
  * `gpu_compute_flux_kernel`: Computes Upwind ($C_0 = 1.0$) and Central ($C_0 = 0.0$) numerical Riemann fluxes with PEC mirror boundary conditions.
  * `gpu_add_flux_kernel`: Lifts numerical surface fluxes into volume residuals using hardware `atomicAdd`.
  * `gpu_inv_mass_kernel`: Multiplies residuals by exact diagonal inverse mass matrix $M^{-1} = 1 / (J w_i w_j w_k)$ (GLL mass lumping property).
  * `gpu_lsrk45_update_kernel`: Vectorized 5-stage Low-Storage Runge-Kutta (Carpenter & Kennedy 1994) state and auxiliary vector update.

### Observation Probes & Diagnostics
* **Sub-Element Spectral Interpolation (`ProbeManager`)**:
  * Collinear and arbitrary 3D observation point probes.
  * Evaluates physical fields at exact arbitrary coordinates using spectral element Lagrange polynomial interpolation ($O(\Delta x^N)$ accuracy) rather than nearest-node rounding.
  * Streams multi-channel time histories (`probe_history.csv`) and JSON metadata (`probes.json`).
* **Continuous Energy Tracking**:
  * Real-time calculation and logging of electromagnetic field energy $\int \frac{1}{2}(\varepsilon |\mathbf{E}|^2 + \mu |\mathbf{H}|^2) d\Omega$.

---

## 2. Numerical Analysis & Verification Suite (`examples/numerical_analysis/`)

A dedicated suite of numerical analysis tools has been implemented to rigorously verify stability, dispersion, and physical wave propagation:

1. **Analytical Matrix Exporter (`analytical_export/`)**:
   * C++ tool hooking into NekWave's `Mesh` class to export reference element mass ($\mathbf{M}$), stiffness ($\mathbf{S}$), and interface flux ($\mathbf{F}_k$) matrices to JSON for any order $N$ and flux $C_0$.
2. **von Neumann Stability Analysis (`neumann_stability/`)**:
   * Maps spatial eigenvalues of the reference element DGTD operator $\mathbf{H}(\mathbf{k})$ scaled by $\Delta t$ into the complex plane.
   * Overlays eigenvalues directly onto the Low-Storage RK45 stability boundary $|G(z)| \le 1$.
   * Automates parameter sweeps across orders $N \in \{2 \dots 10\}$ and fluxes $C_0 \in \{0.0, 0.5, 1.0\}$.
3. **Full Spatio-Temporal Analytical Dispersion (`full_dispersion/`)**:
   * Computes exact coupled discrete dispersion curves folding spatial DGTD matrices into the LSRK45 stability polynomial $G(z)$.
   * Extracts discrete frequency $\omega_{\text{discrete}} = \arg(G(\lambda \Delta t)) / \Delta t$ and phase velocity $v_p = \omega_{\text{discrete}} / k$.
   * Employs eigenvector correlation tracking ($|\mathbf{v}_k^\dagger \mathbf{v}_{k-1}|$) across $k \in [0.01, \pi/\Delta x]$ and propagation angles $\theta \in [0^\circ, 15^\circ, 30^\circ, 45^\circ]$ to eliminate mode hopping.
4. **Two-Probe Multi-Harmonic Numerical Dispersion Benchmark (`numerical_dispersion/`)**:
   * Runs actual 3D CUDA simulations on a periodic channel ($L_x = 16\pi \approx 50.2655\text{ m}$, 64 elements).
   * Simultaneously excites 5 linearly spaced modes from $k = 0.25$ to $3.75\text{ rad/m}$ ($\Delta k = 0.875\text{ rad/m}$). Because $\gcd = 0.125\text{ rad/m}$, all 5 modes have exact integer cycle counts ($n = [2, 9, 16, 23, 30]$), guaranteeing strict $C^\infty$ periodic continuity with zero jump discontinuity.
   * Signal processing pipeline: DC removal, 4-term Blackman-Harris windowing ($-92\text{ dB}$ sidelobe attenuation) to eliminate cross-talk, zero-padded 65,536-bin FFT, and 3-point parabolic peak interpolation for sub-bin frequency precision.
   * Generates dual-panel plots ($v_p/c$ vs. PPW in log scale $628 \to 2$, and $\omega$ vs. $k$ from $0 \to 4$).

---

## 3. Planned Architecture: Element-Major AoSoA (Variant A)

### Planned Memory Layout
To optimize GPU L2 cache locality and reduce memory transaction footprint, the device state will transition from the current Global Component-Major SoA to **Element-Major AoSoA (Variant A)**:

```
Element-Major AoSoA on GPU:
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┬─────┐
│                 Element 0                    │                 Element 1                    │ ... │
│  Ex(Np)   Ey(Np)   Ez(Np)   Hx(Np)   ...     │  Ex(Np)   Ey(Np)   Ez(Np)   Hx(Np)   ...     │     │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┴─────┘
```

* **Data Slicing:**
  $$\text{Memory Shape: } [N_{\text{elt}}][6 \text{ fields}][N_p \text{ GLL points}]$$
  $$\text{Index}(e, \text{field}, i) = e \cdot (6 N_p) + \text{field} \cdot N_p + i$$
* **CPU Host Strategy:**
  * **Keep CPU Host layout unchanged (Standard SoA).**
  * `Case::state_`, `ProbeManager::record()`, initial conditions, and CSV exports continue operating on standard continuous component arrays.
  * In `DgSolver::uploadState()` and `DgSolver::downloadState()`, perform a lightweight GPU packing/unpacking transposition kernel between Host SoA and Device AoSoA.
  * Because zero host-device transfers occur during time stepping, this transposition has **zero runtime overhead** during the simulation.
* **GPU Device Benefits:**
  * In `gpu_volume_curl_kernel`, all 6 field components for element $e$ are stored in a single contiguous memory block of $6 \times N^3 \times 8$ bytes (for $N=8$, $\approx 24.5\text{ KB}$), fitting cleanly into L2 cache and drastically reducing page/TLB thrashing.
  * Consecutive threads $m$ within each element's thread block access consecutive memory locations ($+i$), maintaining $100\%$ warp coalescing.

---

## 4. Hardware Acceleration & Tensor Cores

### Will the Compiler Automatically Run on Tensor Cores?
**No. The CUDA compiler (`nvcc`) will NOT automatically map generic C++ loops or DG tensor contractions to Tensor Cores.**

### Why Tensor Cores Require Explicit Programming:
1. **Target Operation:** Tensor Cores are specialized fixed-function application-specific circuits that exclusively execute small matrix multiply-accumulate (MMA) tile operations of the form:
   $$\mathbf{D} = \mathbf{A} \times \mathbf{B} + \mathbf{C}$$
   Standard loop unrolling (`#pragma unroll`) and scalar arithmetic in `gpu_volume_curl_kernel` compile to standard FP32/FP64 CUDA cores (ALUs), not Tensor Cores.
2. **Explicit APIs Required:** To utilize Tensor Cores, kernels must be explicitly authored using one of the following:
   * **CUDA WMMA API** (`nvcuda::wmma`): Using explicit fragment loaders (`wmma::load_matrix_sync`), matrix multipliers (`wmma::mma_sync`), and storers (`wmma::store_matrix_sync`).
   * **Inline PTX Assembly**: Direct invocation of hardware instructions (e.g., `mma.sync.aligned.m8n8k4.f64`).
   * **NVIDIA CUTLASS**: CUDA C++ template abstractions for high-performance tensor contractions.
   * **cuBLAS / cuTENSOR**: Vendor library calls.
3. **Double Precision (FP64) Tensor Core Availability:**
   * **Datacenter GPUs (A100, H100, B200):** Feature full-rate FP64 Tensor Cores supporting double-precision MMA ($8 \times 8 \times 4$ matrix tiles).
   * **Consumer / GeForce GPUs (RTX 3090, 4090):** Lack high-speed FP64 Tensor Cores; their FP64 throughput is deliberately throttled to $1/64$ of FP32. Tensor Cores on these GPUs only accelerate FP16, BF16, TF32, and INT8.
4. **Dimension Mismatch in High-Order SEM:**
   * Tensor Core tiles require fixed dimensions (e.g., $16 \times 16 \times 16$ for half precision, $8 \times 8 \times 4$ for double precision).
   * In DG-SEM, the 1D contraction matrix $\mathbf{D}$ is $N \times N$, where typical polynomial orders are $N \in [2, 10]$.
   * To achieve high Tensor Core efficiency, elements must be batched together (e.g., batching multiple elements into a single larger GEMM) using specialized libraries such as libCEED or custom batched WMMA kernels.

---

## 5. Architectural Plan: Boundary Conditions (PEC, PMC, Periodic), NekCEM Mesh & HDF5 Output

### Objective
Provide full, user-configurable boundary condition support (**PEC**, **PMC**, **Periodic**), full NekCEM `.rea` mesh compatibility, an automated verification test harness, and high-performance HDF5 field export. (PML is explicitly deferred to a later milestone).

```
Boundary Condition Classification:
┌─────────────────────┬───────────────────────────┬────────────────────────────────────────────┐
│ Boundary Type       │ Field Physics Constraint  │ Ghost State Jump Formulation               │
├─────────────────────┼───────────────────────────┼────────────────────────────────────────────┤
│ PEC (Electric Wall) │ n x E = 0,  n . H = 0     │ [[E]] = -2 E^-,  [[H]] = 0                 │
│ PMC (Magnetic Wall) │ n x H = 0,  n . E = 0     │ [[H]] = -2 H^-,  [[E]] = 0                 │
│ Periodic (Wrap)     │ E^+ = E_opp^-, H^+ = H_opp^- │ Evaluates standard internal Riemann flux │
└─────────────────────┴───────────────────────────┴────────────────────────────────────────────┘
```

---

### Step 1: NekCEM Mesh Loader & Boundary Tag Mapping (`src/mesh.cpp`)

NekCEM encodes boundary conditions on all 6 quadrilateral faces of every hexahedral element in the `BOUNDARY CONDITIONS` section of the `.rea` file.

1. **Tag Standardizer**:
   Extend `Mesh::loadFromRea` to parse 3-character face tags into an internal enum:
   ```cpp
   enum class BoundaryType : int {
       INTERNAL = 0,
       PERIODIC = 0,  // Handled by pairing volIdxPlus across periodic faces
       PEC      = 1,  // Perfect Electric Conductor
       PMC      = 2   // Perfect Magnetic Conductor / Symmetry
   };
   ```
   * Tag `W  ` (Wall) $\longrightarrow$ `BoundaryType::PEC`
   * Tag `SYM` or `M  ` (Symmetry / Magnetic) $\longrightarrow$ `BoundaryType::PMC`
   * Tag `P  ` (Periodic) $\longrightarrow$ `BoundaryType::PERIODIC` (resolves neighbor element ID and face ID pairs)
   * Tag `E  ` (Element) $\longrightarrow$ `BoundaryType::INTERNAL`
2. **Configurable Box Mesh Generator (`createBoxMesh`)**:
   Add explicit plane-by-plane boundary specification in `Mesh::createBoxMesh` and `Config`:
   ```ini
   # In case parameter file (.par):
   bc_xmin = PEC
   bc_xmax = PMC
   bc_ymin = PERIODIC
   bc_ymax = PERIODIC
   bc_zmin = PEC
   bc_zmax = PEC
   ```
   Allows users to define custom mixed waveguide, cavity, and periodic problems without external `.rea` files.

---

### Step 2: GPU Numerical Riemann Flux Dispatch (`src/device/dg_kernels.cu`, `src/dg_solver.cpp`)

1. **Replace Binary Flag with BC Tag Array**:
   * Currently, `dg_solver.cpp` uploads a binary `d_isPEC_` buffer.
   * Generalize to `int* d_bcType_` of length `totalFacePoints`, populated from `fd.bcType`:
     * `0`: Internal or Periodic face (`volIdxPlus >= 0`)
     * `1`: PEC face (`volIdxPlus = -1`)
     * `2`: PMC face (`volIdxPlus = -1`)
2. **GPU Kernel Update (`gpu_compute_flux_kernel`)**:
   ```cuda
   int bc = bcType[p];
   double dEx, dEy, dEz, dHx, dHy, dHz;

   if (bc == 1) {
       // PEC mirror: n x E^+ = -n x E^- => [[E]] = -2 E^-, [[H]] = 0
       dEx = -2.0 * Ex_m;  dEy = -2.0 * Ey_m;  dEz = -2.0 * Ez_m;
       dHx =  0.0;         dHy =  0.0;         dHz =  0.0;
   } else if (bc == 2) {
       // PMC mirror: n x H^+ = -n x H^- => [[H]] = -2 H^-, [[E]] = 0
       dEx =  0.0;         dEy =  0.0;         dEz =  0.0;
       dHx = -2.0 * Hx_m;  dHy = -2.0 * Hy_m;  dHz = -2.0 * Hz_m;
   } else {
       // Internal interface or Periodic boundary (connected via volIdxPlus)
       int vP = volIdxPlus[p];
       dEx = state[0 * npts + vP] - Ex_m;
       dEy = state[1 * npts + vP] - Ey_m;
       dEz = state[2 * npts + vP] - Ez_m;
       dHx = state[3 * npts + vP] - Hx_m;
       dHy = state[4 * npts + vP] - Hy_m;
       dHz = state[5 * npts + vP] - Hz_m;
   }
   ```
   The rest of the flux calculation (cross products $\hat{n} \times [[\mathbf{E}]]$, $\hat{n} \times [[\mathbf{H}]]$, double cross penalty terms, and lifting via `gpu_add_flux_kernel`) remains mathematically exact and invariant!

---

### Step 3: Verification Test Harness (`tests/nekwave_test.cpp`)

To guarantee rock-solid stability and correctness, add dedicated tests to `nekwave-test`:

1. **`NekCemMeshLoaderTest`**:
   * Reads a verified benchmark mesh with mixed boundary tags (`W  `, `SYM`, `P  `, `E  `).
   * Validates element corners, face outward normals $\hat{n}$, metric surface areas $dA$, and boundary tag assignments against expected ground truth.
2. **`MixedBCCavityTest` (PEC + PMC + Periodic)**:
   * Sets up a box with Periodic in $X$, PMC in $Y$, and PEC in $Z$.
   * Excites a known analytical standing wave mode:
     $$E_z(x, y, z, 0) = \cos\left(\frac{2\pi m x}{L_x}\right) \cos\left(\frac{\pi n y}{L_y}\right) \sin\left(\frac{\pi p z}{L_z}\right)$$
   * Verifies that:
     1. Natural boundary conditions ($\hat{n} \times \mathbf{E} = 0$ on PEC, $\hat{n} \times \mathbf{H} = 0$ on PMC) are preserved weakly with zero boundary reflection error.
     2. Energy is strictly conserved ($dU/dt = 0$ for Central flux, $dU/dt \le 0$ for Upwind flux).
     3. Discrete oscillation frequency matches the analytical dispersion frequency $\omega = c \sqrt{k_x^2 + k_y^2 + k_z^2}$.

---

### Step 4: High-Performance HDF5 Output Pipeline

Replace heavy ASCII CSV exports (`field_final.csv`) with a high-performance binary **HDF5 / XDMF** writer:

1. **Architecture (`src/io/hdf5_writer.hpp`, `src/io/hdf5_writer.cpp`)**:
   * Optional dependency handled via CMake: `find_package(HDF5 COMPONENTS C)`.
   * Directly writes IEEE-754 binary double-precision datasets:
     * `/mesh/coordinates`: $(N_{\text{pts}}, 3)$ nodal positions $(x, y, z)$.
     * `/time_series/{step}/E`: $(N_{\text{pts}}, 3)$ vector field $(E_x, Ey, Ez)$.
     * `/time_series/{step}/H`: $(N_{\text{pts}}, 3)$ vector field $(H_x, Hy, Hz)$.
     * Attributes: `time`, `step`, `order_N`, `num_elements`, `cfl`.
2. **XDMF XML Descriptor (`.xmf`)**:
   * Automatically generates an accompanying lightweight XML metadata file.
   * Enables native, zero-copy visualization of 3D high-order polynomial solutions directly in **ParaView** and **VisIt**.

---

## 6. Completed Milestone: NekCEM Mesh Ingestion & 3D Box PEC Verification

### 1. Direct NekCEM `.rea` Mesh Parsing (`src/mesh.cpp`)
* **Mesh Ingestion**: Implemented reading of legacy Nekton / NekCEM / Nek5000 `.rea` mesh files directly from `contrib/NekCEM/tests/3dboxpec/3dboxpec.rea` (27 elements in $3 \times 3 \times 3$ grid spanning $[-1, 1]^3$).
* **Boundary Condition Tag Normalization**:
  * `PEC`, `W`, `v` $\rightarrow$ Perfect Electric Conductor ($[[\mathbf{E}]] = -2\mathbf{E}^-$, $[[\mathbf{H}]] = \mathbf{0}$).
  * `PMC`, `SYM`, `s` $\rightarrow$ Perfect Magnetic Conductor ($[[\mathbf{H}]] = -2\mathbf{H}^-$, $[[\mathbf{E}]] = \mathbf{0}$).
  * `P`, `p`, `PERIODIC` $\rightarrow$ Periodic coordinate wrapping across boundary faces.
  * `E` $\rightarrow$ Internal element interfaces.
* **Face Trace Node Matching**: Precomputes Nanson metric areas $dA$ and outward unit normals $\hat{n}$. Resolves interface nodes across element boundaries via 3D physical coordinate coincidence matching ($< 10^{-10}$).

### 2. Unified GPU Boundary Flux Kernel (`src/device/dg_kernels.cu`, `src/dg_solver.cpp`)
* Generalized `gpu_compute_flux_kernel` with integer dispatch `bcType` (`0: INTERNAL/PERIODIC`, `1: PEC`, `2: PMC`).
* Preserves zero tangential electric field on PEC and zero tangential magnetic field on PMC.
* Seamlessly uploads `d_bcType_` to GPU without altering host SoA memory structures.

### 3. Canonical 3D Box PEC Benchmark (`examples/3dboxpec/`, `tests/input/3dboxpec.par`)
* **Initial Condition**: Exact standing wave eigenmode from `contrib/NekCEM/tests/3dboxpec/3dboxpec.usr`:
  $$\mathbf{E}(t=0) = \mathbf{0}$$
  $$H_x = -\sin(\pi x)\cos(\pi y)\cos(\pi z)/\sqrt{6}$$
  $$H_y = -\cos(\pi x)\sin(\pi y)\cos(\pi z)/\sqrt{6}$$
  $$H_z =  2\cos(\pi x)\cos(\pi y)\sin(\pi z)/\sqrt{6}$$
* **Analytical Solution**: Oscillation frequency $\omega = \pi \sqrt{3} \approx 5.44140\,\text{rad/s}$.
* **Test Suite Integration**:
  * `NekCemReaMeshTest` in `tests/nekwave_test.cpp`: Direct unit verification of mesh reading, face counts (54 PEC, 108 internal E), 50-step GPU simulation, energy conservation ($0.08\%$ relative diff), and analytical mode accuracy.
  * `BoxPec3DTests` in CTest: Standalone 100-step test case with relative $L_2$ error $< 0.05$.
  * **CTest Status**: 10 out of 10 tests passing ($100\%$).

---

## 7. Completed Milestone: High-Performance HDF5 & XDMF Output Pipeline

### 1. Dual-Layer HDF5 / XDMF Field Architecture (`src/hdf5_writer.hpp`, `src/hdf5_writer.cpp`)
* **Structured Data Hierarchy**:
  * `/mesh/coordinates`: $(N_{\text{pts}}, 3)$ nodal coordinates $(x, y, z)$ in double precision.
  * `/time_series/step_{N}/E`: $(N_{\text{pts}}, 3)$ electric vector field $(E_x, E_y, E_z)$ at each output time step.
  * `/time_series/step_{N}/H`: $(N_{\text{pts}}, 3)$ magnetic vector field $(H_x, H_y, H_z)$ at each output time step.
  * Dataset & Group Attributes: `time`, `step`, `order_N`, `num_elements`, `total_points`.
* **Zero-Copy ParaView / VisIt Visualization (`.xmf`)**:
  * Emits an accompanying XML descriptor conforming to the XDMF 3.0 specification (`TopologyType="Polyvertex"`, `GeometryType="XYZ"`).
  * Allows direct time series visualization, streamlines, vector glyphs, and slicing in ParaView or VisIt without post-processing plugins.
* **Portable IEEE-754 Binary Fallback**:
  * When native `libhdf5` is not detected at compile time, NekWave automatically switches to an optimized binary format (`NEKWAVE_BIN_V1`) without crashing or requiring external libraries.
  * Emits an accompanying `scripts/convert_to_hdf5.py` tool to convert `.bin` files into standard HDF5 `.h5` files with full metadata.

### 2. Configuration & Lifecycle Integration
* Added parameters to `Config` (`src/config.hpp`):
  * `export_fields = true/false`
  * `export_format = hdf5` (or `csv`, `binary`, `auto`)
  * `export_freq = N` (periodic dumping interval during `simulate()`)
* Integrated `Hdf5Writer` directly into the `Case` lifecycle (`src/case.cpp`): initializes and records step 0 in `preprocess()`, streams field snapshots at regular intervals in `simulate()`, and cleanly finalizes the archive in `postprocess()`.

### 3. Verification & Test Suite
* Added `Hdf5WriterTest` to `tests/nekwave_test.cpp`, verifying coordinate dumping, vector field layout, XDMF XML validity, and file size consistency.
* Fully verified with 10 out of 10 tests passing in `ctest`.

---

## 8. Completed Milestone: Dual Termination Criteria & Decoupled I/O Frequencies

### 1. Robust Simulation Termination Criteria (`final_time` & `max_steps`)
* **Flexible Termination Logic**:
  * `final_time`: Target physical simulation end time. Time stepper dynamically adapts the final time step ($\Delta t_{\text{last}} = t_{\text{final}} - t_{\text{current}}$) to land on $t_{\text{final}}$ with machine precision ($< 10^{-14}$) without overshooting.
  * `max_steps`: Hard upper bound on step iterations (aliases: `num_steps`, `steps`).
  * **Dual Termination Check**: When both parameters are provided, simulation terminates safely as soon as either criterion is satisfied:
    $$\text{terminate if } (t \ge t_{\text{final}}) \lor (n_{\text{step}} \ge n_{\text{max}})$$
* **Single Criteria Fallback**: If only `final_time` is specified, `max_steps` defaults to unlimited; if only `max_steps` is specified, `final_time` is disabled.

### 2. Decoupled Diagnostic vs. Field Snapshot Frequencies
* Decoupled runtime logging from disk writing to avoid I/O bottlenecks:
  * `output_frequency` (alias `output_freq`): Controls console diagnostic logging and scalar energy history (`energy_history.csv`).
  * `save_frequency` (alias `save_freq`, `export_freq`): Controls heavy 3D electromagnetic volume field snapshots exported to HDF5 (`fields.h5` and `fields.xmf`).
  * Both frequencies operate completely independently during `simulate()`.
  * The final simulation state is always guaranteed to be recorded and saved.

### 3. Verification & Unit Tests
* Added `TerminationCriteriaTest` in `tests/nekwave_test.cpp` verifying:
  1. Termination by `final_time` alone.
  2. Termination by `max_steps` alone.
  3. Combined check with `max_steps` triggering first.
  4. Combined check with `final_time` triggering first.
* Added `DecoupledFrequencyTest` in `tests/nekwave_test.cpp` verifying independent snapshot recording vs. console logging.


