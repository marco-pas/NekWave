# Mathematical and Algorithmic Methodology: NekCEM Time-Domain Maxwell Solver

This document provides a comprehensive technical breakdown of the numerical methodology implemented in `NekCEM`'s core files:
- `cem_drive.F`: Driver, initialization, time stepping loop, and execution pipeline.
- `cem_maxwell.F`: Discontinuous Galerkin Spectral Element Method (DG-SEM) spatial operator for 3D/2D Maxwell equations.

---

## 1. Governing Equations

The solver targets the time-domain Maxwell curl equations in first-order form:

$$\varepsilon \frac{\partial \mathbf{E}}{\partial t} = \nabla \times \mathbf{H} - \sigma \mathbf{E} - \mathbf{J}$$

$$\mu \frac{\partial \mathbf{H}}{\partial t} = -\nabla \times \mathbf{E} - \mathbf{M}$$

where:
- $\mathbf{E} = (E_x, E_y, E_z)^T$ is the electric field.
- $\mathbf{H} = (H_x, H_y, H_z)^T$ is the magnetic field.
- $\varepsilon(\mathbf{x}), \mu(\mathbf{x})$ are the electric permittivity and magnetic permeability.
- $\sigma(\mathbf{x})$ is electrical conductivity (loss).
- $\mathbf{J}, \mathbf{M}$ are electric and magnetic source currents.

In lossless linear media ($\sigma = 0, \mathbf{J} = \mathbf{0}, \mathbf{M} = \mathbf{0}$):

$$\frac{\partial \mathbf{E}}{\partial t} = \frac{1}{\varepsilon} (\nabla \times \mathbf{H}), \quad \frac{\partial \mathbf{H}}{\partial t} = -\frac{1}{\mu} (\nabla \times \mathbf{E})$$

---

## 2. Spatial Discretization: DG Spectral Element Method (DG-SEM)

The computational domain $\Omega$ is partitioned into non-overlapping conforming hexahedral elements $\Omega^e$ ($e = 1, \dots, E$).

### 2.1 Collocation on Gauss-Lobatto-Legendre (GLL) Nodes
Each element is mapped to a reference cube $[-1, 1]^3$ via the coordinate mapping:

$$\mathbf{x} = \mathbf{x}^e(r, s, t), \quad (r, s, t) \in [-1, 1]^3$$

Fields are represented using a nodal polynomial basis of degree $P = N - 1$ in each direction (with $N$ GLL points per direction, so each element contains $N_p = N^3$ nodes):

$$\mathbf{u}^e(r, s, t) \approx \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} \sum_{k=0}^{N-1} \mathbf{u}_{ijk}^e \, \ell_i(r) \ell_j(s) \ell_k(t)$$

where $\ell_i(\xi)$ are 1D Lagrange interpolating polynomials through the GLL nodes $\{\xi_i\}_{i=0}^{N-1}$.

### 2.2 Diagonal Mass Matrix Property
NekCEM uses the GLL quadrature rule with the same nodes as the polynomial interpolation:

$$\int_{-1}^1 f(\xi) \, d\xi \approx \sum_{i=0}^{N-1} f(\xi_i) \, \rho_i$$

Because Lagrange polynomials satisfy the discrete orthogonality:

$$\sum_{k=0}^{N-1} \ell_i(\xi_k) \ell_j(\xi_k) \rho_k = \rho_i \delta_{ij}$$

the element mass matrix is **strictly diagonal**:

$$\mathbf{M}^e_{ijk, i'j'k'} = J_{ijk}^e \, \rho_i \rho_j \rho_k \, \delta_{ii'} \delta_{jj'} \delta_{kk'}$$

where $J_{ijk}^e$ is the coordinate transformation Jacobian determinant. **Inverting the mass matrix is therefore trivial $O(N_p)$ point-wise division.**

---

## 3. Step-by-Step Algorithmic Pipeline

The code execution proceeds in three main tiers:
1. **Setup & Initialization** (`cem_drive.F` $\to$ `cem_init`)
2. **Main Time Stepping Loop** (`cem_drive.F` $\to$ `time_advancing_pde`)
3. **Spatial Operator Evaluation** (`cem_maxwell.F` $\to$ `cem_maxwell_op_rk` / `cem_maxwell_op`)

```
========================================================================
[1. cem_init] (cem_drive.F)
    ├── Read input parameters (.rea / .par)
    ├── Generate mesh coordinates & topology (gengeom)
    ├── Compute GLL points, weights & 1D derivative matrix D (load_semhat_weighted / DGLL)
    ├── Compute metric terms (rx, sx, tx, ...) and Jacobians
    ├── Compute face normal vectors (unx, uny, unz) and face areas
    └── Initialize RK storage coefficients (rk_storage)
========================================================================
[2. cem_solve] (cem_drive.F)
    ├── Compute CFL-restricted time step dt (get_dxmin, set_dt)
    ├── Set initial conditions (cem_maxwell_init, cem_maxwell_init_fields)
    └── Call time_advancing_pde
========================================================================
[3. time_advancing_pde] (cem_drive.F)
    └── Loop over timesteps: istep = 1 ... nsteps:
         └── cem_maxwell_op_rk (cem_maxwell.F)
              ├── Loop over 5 Runge-Kutta stages (rkstep = 1 ... 5):
              │    ├── 1. Update sub-stage time: rk_c(rkstep)
              │    ├── 2. Evaluate Spatial Operator: cem_maxwell_op
              │    │      ├── A. Volume Term: cem_maxwell -> maxwell_wght_curl
              │    │      │      ├── local_grad3 (tensor contractions via D, D^T)
              │    │      │      ├── Apply 3D quadrature weights (w3mn)
              │    │      │      └── Metric mapping to physical curl components
              │    │      ├── B. Trace Extraction: cem_maxwell_restrict_to_face
              │    │      ├── C. Numerical Flux: cem_maxwell_flux (Upwind / Central)
              │    │      │      ├── Parallel face exchange / communication (gs_op_fields)
              │    │      │      ├── Physical boundary conditions (PEC, PML, ABC)
              │    │      │      └── Form flux jumps [[E]], [[H]]
              │    │      ├── D. Surface Lifting: cem_maxwell_add_flux_to_res
              │    │      └── E. Inverse Mass Scaling: cem_maxwell_invqmass
              │    └── 3. Accumulate State: rk_maxwell_ab (rk4_upd)
              │           ├── k = a_s * k + dt * res
              │           └── u = u + b_s * k
              └── Optional filter: q_filter
========================================================================
[4. cem_end] (cem_drive.F)
    └── Print timing benchmarks, diagnostics, and finalize
```

---

## 4. Deep Dive into the Spatial RHS (`cem_maxwell_op`)

In each RK stage, the subroutine `cem_maxwell_op` evaluates:

$$\text{RHS} = \mathbf{M}^{-1} \left( \mathbf{K} \, \mathbf{u} + \mathbf{F}^* \right)$$

### 4.1 Volume Differentiation (`maxwell_wght_curl` & `local_grad3`)

To evaluate the curl in physical space, derivatives with respect to reference coordinates $(r, s, t)$ are computed first using 1D derivative matrix $\mathbf{D} \in \mathbb{R}^{N \times N}$ and its transpose $\mathbf{D}^T$:

$$\mathbf{u}_r = (\mathbf{I} \otimes \mathbf{I} \otimes \mathbf{D}) \mathbf{u}$$

$$\mathbf{u}_s = (\mathbf{I} \otimes \mathbf{D} \otimes \mathbf{I}) \mathbf{u}$$

$$\mathbf{u}_t = (\mathbf{D} \otimes \mathbf{I} \otimes \mathbf{I}) \mathbf{u}$$

In `local_grad3` (`nek5_grad.F`), this is done efficiently via matrix-matrix products:
1. `mxm(D, N, u, N, ur, N*N)`: Contraction along $r$.
2. Slice-wise loop over $k$: `mxm(u(:,:,k), N, Dt, N, us(:,:,k), N)`: Contraction along $s$.
3. `mxm(u, N*N, Dt, N, ut, N)`: Contraction along $t$.

The reference derivatives are then scaled by the 3D quadrature weights $W_{ijk} = \rho_i \rho_j \rho_k$ (`w3mn`) and mapped to physical space via the metric terms:

$$\begin{aligned}
(\nabla \times \mathbf{u})_x^w &= (u_{3,r}^w r_y + u_{3,s}^w s_y + u_{3,t}^w t_y) - (u_{2,r}^w r_z + u_{2,s}^w s_z + u_{2,t}^w t_z) \\
(\nabla \times \mathbf{u})_y^w &= (u_{1,r}^w r_z + u_{1,s}^w s_z + u_{1,t}^w t_z) - (u_{3,r}^w r_x + u_{3,s}^w s_x + u_{3,t}^w t_x) \\
(\nabla \times \mathbf{u})_z^w &= (u_{2,r}^w r_x + u_{2,s}^w s_x + u_{2,t}^w t_x) - (u_{1,r}^w r_y + u_{1,s}^w s_y + u_{1,t}^w t_y)
\end{aligned}$$

For Maxwell:
- **$\mathbf{E}$-residual volume term**: $\mathbf{w}_E = \nabla \times \mathbf{H}$
- **$\mathbf{H}$-residual volume term**: $\mathbf{w}_H = -\nabla \times \mathbf{E}$

---

### 4.2 Face Traces and Numerical Fluxes (`cem_maxwell_flux3d`)

At element interfaces, fields are discontinuous. The interface coupling is enforced weakly through a numerical flux $\mathbf{F}^*$.

1. **Trace Extraction**: `cem_maxwell_restrict_to_face` extracts the field values from the element volume nodes to the $2 \times d$ faces of each hex element.
2. **Jump Calculation**: Let $\mathbf{u}^-$ be the interior trace and $\mathbf{u}^+$ be the neighbor (exterior) trace. Jumps across the interface with outward normal $\hat{n} = \hat{n}^-$ are defined as:
   $$[[\mathbf{E}]] = \mathbf{E}^+ - \mathbf{E}^-, \quad [[\mathbf{H}]] = \mathbf{H}^+ - \mathbf{H}^-$$
3. **Upwind Numerical Flux Formulation**:
   Using local characteristic admittance $Y = \sqrt{\varepsilon / \mu}$ and impedance $Z = \sqrt{\mu / \varepsilon}$, with averages $Y_0 = \frac{Y^- + Y^+}{2}$, $Z_0 = \frac{Z^- + Z^+}{2}$:
   $$\mathbf{F}^*_E = -\frac{Y^+}{2 Y_0} (\hat{n} \times [[\mathbf{H}]]) - \frac{C_0}{2 Y_0} \hat{n} \times (\hat{n} \times [[\mathbf{H}]]) \quad (\text{or with jumps in } [[\mathbf{E}]])$$
   $$\mathbf{F}^*_H = \frac{Z^+}{2 Z_0} (\hat{n} \times [[\mathbf{E}]]) - \frac{C_0}{2 Z_0} \hat{n} \times (\hat{n} \times [[\mathbf{E}]])$$
   where $C_0 = 1.0$ yields the full **Rankine-Hugoniot upwind flux**, and $C_0 = 0.0$ yields the **central flux**.
4. **Boundary Conditions**:
   - **PEC (Perfect Electric Conductor)**: $\hat{n} \times \mathbf{E} = 0 \implies \mathbf{E}^+ = -\mathbf{E}^-$, $\mathbf{H}^+ = \mathbf{H}^-$.
   - **PMC (Perfect Magnetic Conductor)**: $\hat{n} \times \mathbf{H} = 0 \implies \mathbf{H}^+ = -\mathbf{H}^-$, $\mathbf{E}^+ = \mathbf{E}^-$.
   - **Silver-Müller Absorbing Boundary (ABC)**: Exterior fields set to zero incoming radiation.

---

### 4.3 Surface Lifting (`cem_maxwell_add_flux_to_res`)

The numerical surface flux vector $\mathbf{F}^*$ on each face is multiplied by the face quadrature metric area $A_j$ (`aream`) and added directly into the corresponding boundary nodal positions of the volume residual vector (`resEN`, `resHN`).

---

### 4.4 Mass Matrix Inversion (`cem_maxwell_invqmass`)

Because the mass matrix is diagonal, each volume node $i$ of the accumulated residual is simply divided by its mass entry:

$$\text{resEN}_{i, c} \leftarrow \text{resEN}_{i, c} \times \frac{1}{\varepsilon_i J_i W_i}$$

$$\text{resHN}_{i, c} \leftarrow \text{resHN}_{i, c} \times \frac{1}{\mu_i J_i W_i}$$

for components $c \in \{1, 2, 3\}$.

---

## 5. Time Stepping: Low-Storage RK45 (`cem_maxwell_op_rk`)

NekCEM employs a 5-stage, 4th-order Low-Storage Runge-Kutta (LSRK45) scheme (Carpenter & Kennedy 1994). 

### Advantages:
- Only **one auxiliary memory vector** $\mathbf{k}$ is required per field variable (drastically reduces RAM compared to standard Butcher-tableau RK4).
- High CFL stability limit ($CFL \approx 1.7$–$2.2$ depending on polynomial order).

### Update Equation (for each stage $s = 1, \dots, 5$):
$$\mathbf{k} \leftarrow a_s \, \mathbf{k} + \Delta t \, \text{RHS}(\mathbf{u}, t + c_s \Delta t)$$

$$\mathbf{u} \leftarrow \mathbf{u} + b_s \, \mathbf{k}$$

### Coefficients (from `rk_storage` in `cem_common.F`):

| Stage $s$ | $a_s$ | $b_s$ | $c_s$ |
|---|---|---|---|
| **1** | $0$ | $\frac{1432997174477}{9575080441755}$ | $0$ |
| **2** | $-\frac{567301805773}{1357537059087}$ | $\frac{5161836677717}{13612068292357}$ | $\frac{1432997174477}{9575080441755}$ |
| **3** | $-\frac{2404267990393}{2016746695238}$ | $\frac{1720146321549}{2090206949498}$ | $\frac{2526269341429}{6820363962896}$ |
| **4** | $-\frac{3550918686646}{2091501179385}$ | $\frac{3134564353537}{4481467310338}$ | $\frac{2006345519317}{3224310063776}$ |
| **5** | $-\frac{1275806237668}{842570457699}$ | $\frac{2277821191437}{14882151754819}$ | $\frac{2802321613138}{2924317926251}$ |

---

## 6. Affine Transformations in Hexahedral DG-SEM

### 6.1 Where Affine Transformations Are Used
Affine transformations belong in the **`Mesh` geometry initialization** (`Mesh::initialize`, `Mesh::setAffineBox`, or `gengeom` in NekCEM).

### 6.2 Mathematical Definition
When an element $\Omega^e$ is an affine brick or parallelepiped, the mapping from the reference cube $[-1, 1]^3$ to physical space $\mathbf{x} = (x, y, z)$ is linear:

$$\mathbf{x}(\boldsymbol{\xi}) = \mathbf{x}_0 + \mathcal{J}_{\text{aff}} \boldsymbol{\xi}, \quad \text{where } \boldsymbol{\xi} = (r, s, t)^T$$

For an axis-aligned bounding box $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}] \times [z_{\min}, z_{\max}]$:

$$\begin{aligned}
x(r) &= \frac{x_{\min} + x_{\max}}{2} + \frac{\Delta x}{2} r \\
y(s) &= \frac{y_{\min} + y_{\max}}{2} + \frac{\Delta y}{2} s \\
z(t) &= \frac{z_{\min} + z_{\max}}{2} + \frac{\Delta z}{2} t
\end{aligned}$$

### 6.3 Metric Invariants for Affine Elements
The Jacobian matrix $\mathcal{J}_{\text{aff}}$ and its determinant $J$ are **constant throughout the entire element**:

$$\mathcal{J}_{\text{aff}} = \begin{bmatrix} \frac{\Delta x}{2} & 0 & 0 \\ 0 & \frac{\Delta y}{2} & 0 \\ 0 & 0 & \frac{\Delta z}{2} \end{bmatrix}, \quad J = \det(\mathcal{J}_{\text{aff}}) = \frac{\Delta x \, \Delta y \, \Delta z}{8}$$

The metric derivatives $\mathcal{J}^{-T}$ are similarly constant across all GLL points:

$$\begin{bmatrix} r_x & s_x & t_x \\ r_y & s_y & t_y \\ r_z & s_z & t_z \end{bmatrix} = \begin{bmatrix} \frac{2}{\Delta x} & 0 & 0 \\ 0 & \frac{2}{\Delta y} & 0 \\ 0 & 0 & \frac{2}{\Delta z} \end{bmatrix}$$

**Performance Consequence**: For affine elements, memory bandwidth can be drastically reduced because metric terms do not need to be stored per-point; a single set of 9 values and one Jacobian suffices for all $N_p$ nodes in that element.

---

## 7. Piola Transformations in Computational Electromagnetics

### 7.1 Where the Piola Transformation Belongs
The Piola transformation belongs in the **`Physics` spatial operator**:
1. **In Volume Integration (`compute_weighted_curl`)**: Mapping curls between reference and physical coordinates.
2. **In Surface Flux Evaluation (`computeFlux`)**: Preserving normal flux continuity across element interfaces via Nanson's formula.

### 7.2 Differential Forms and Piola Mappings
In CEM, electromagnetic fields are differential forms with specific geometric conservation properties:
* **Electric/Magnetic field intensities ($\mathbf{E}, \mathbf{H}$)** are **1-forms** ($H(\text{curl})$). They must preserve tangential continuity across interfaces ($\hat{n} \times [[\mathbf{E}]] = 0$).
* **Flux densities ($\mathbf{D}, \mathbf{B}$)** and **curls ($\nabla \times \mathbf{E}, \nabla \times \mathbf{H}$)** are **2-forms** ($H(\text{div})$). They must preserve normal continuity across interfaces ($\hat{n} \cdot [[\mathbf{B}]] = 0$).

#### 1. Covariant Piola Transformation (for 1-forms $\mathbf{E}, \mathbf{H}$):
$$\hat{\mathbf{u}}(\boldsymbol{\xi}) = \mathcal{J}^T \mathbf{u}(\mathbf{x}) \iff \mathbf{u}(\mathbf{x}) = \mathcal{J}^{-T} \hat{\mathbf{u}}(\boldsymbol{\xi})$$
This guarantees circulation invariance:

$$\int_C \mathbf{u} \cdot d\mathbf{x} = \int_{C_{\text{ref}}} \hat{\mathbf{u}} \cdot d\boldsymbol{\xi}$$

#### 2. Contravariant Piola Transformation (for 2-forms $\mathbf{B}, \mathbf{D}$, and curls):
$$\tilde{\mathbf{w}}(\boldsymbol{\xi}) = J \, \mathcal{J}^{-1} \mathbf{w}(\mathbf{x}) \iff \mathbf{w}(\mathbf{x}) = \frac{1}{J} \mathcal{J} \, \tilde{\mathbf{w}}(\boldsymbol{\xi})$$
This guarantees surface flux invariance:

$$\int_S \mathbf{w} \cdot d\mathbf{S} = \int_{S_{\text{ref}}} \tilde{\mathbf{w}} \cdot d\hat{\mathbf{S}}$$

and automatically commutes with the divergence operator: $\nabla_{\mathbf{x}} \cdot \mathbf{w} = \frac{1}{J} \nabla_{\boldsymbol{\xi}} \cdot \tilde{\mathbf{w}}$.

### 7.3 Commutation with the Curl Operator
When a 1-form field is transformed covariantly ($\hat{\mathbf{u}} = \mathcal{J}^T \mathbf{u}$), its curl transforms contravariantly:

$$\nabla_{\mathbf{x}} \times \mathbf{u} = \frac{1}{J} \, \mathcal{J} \left( \nabla_{\boldsymbol{\xi}} \times \hat{\mathbf{u}} \right)$$

In NekCEM, rather than transforming the entire field to reference space and back, the code computes derivatives of the Cartesian components $(u_x, u_y, u_z)$ directly via the metric cofactor relations ($J r_x, J s_x, \dots$), which satisfies the discrete **Piola identity**:

$$\nabla_{\boldsymbol{\xi}} \cdot \left( J \, \mathcal{J}^{-1} \right) = \mathbf{0}$$

### 7.4 Surface Fluxes: Nanson's Formula
At element faces, a physical area element $d\mathbf{S} = \hat{n} \, dA$ is mapped from reference face normal $d\hat{\mathbf{S}} = \hat{n}_{\text{ref}} \, dA_{\text{ref}}$ via **Nanson's formula**:

$$\hat{n} \, dA = J \, \mathcal{J}^{-T} \hat{n}_{\text{ref}} \, dA_{\text{ref}}$$

In `cem_maxwell_flux3d`, this relation ensures that tangential traces and jump terms $\hat{n} \times [[\mathbf{E}]]$ and $\hat{n} \times [[\mathbf{H}]]$ computed in physical coordinates correctly couple with the face metric areas $A_j$ (`aream`) during surface flux lifting (`cem_maxwell_add_flux_to_res`).

---

## 8. NekCEM Subroutine Reference Map

| Subroutine | Source File | Mathematical / Computational Function |
|---|---|---|
| `cem_init` | `cem_drive.F` | Initializes geometry, parameters, GLL operators, and RK constants |
| `time_advancing_pde` | `cem_drive.F` | Master time loop across all timesteps |
| `cem_maxwell_op_rk` | `cem_maxwell.F` | Loops over the 5 stages of the LSRK45 method |
| `rk_c` | `cem_common.F` | Computes the sub-stage time $t_{stage} = t + c_s \Delta t$ |
| `cem_maxwell_op` | `cem_maxwell.F` | Orchestrates the spatial RHS evaluation (volume + face + mass inversion) |
| `cem_maxwell` | `cem_maxwell.F` | Dispatches volume curl calculation |
| `maxwell_wght_curl` | `cem_maxwell.F` | Computes metric-weighted physical curl $(\nabla \times \mathbf{u})^w$ |
| `local_grad3` | `nek5_grad.F` | Tensor-product contractions computing reference derivatives $(u_r, u_s, u_t)$ |
| `mxm` | `nek5_mxm_wrapper.F`| Fast column-major matrix multiplication $C = A \cdot B$ |
| `cem_maxwell_restrict_to_face` | `cem_maxwell.F` | Extracts boundary traces of $\mathbf{E}, \mathbf{H}$ on element faces |
| `gs_op_fields` | `comm_mpi.F` | Inter-element communication & trace exchange (gather-scatter) |
| `cem_maxwell_flux` / `flux3d` | `cem_maxwell.F` | Computes upwind / central numerical fluxes using jumps $[[\mathbf{E}]], [[\mathbf{H}]]$ |
| `cem_maxwell_flux_pec` | `cem_maxwell.F` | Applies Perfect Electric Conductor boundary condition |
| `cem_maxwell_add_flux_to_res` | `cem_maxwell.F` | Lifts face fluxes into the volume residual nodes |
| `cem_maxwell_invqmass` | `cem_maxwell.F` | Inverts diagonal mass matrix (divides by $\varepsilon J W$ and $\mu J W$) |
| `rk_maxwell_ab` / `rk4_upd` | `cem_common.F` / `cem_maxwell.F` | Applies low-storage Runge-Kutta accumulator updates |
| `q_filter` | `filter.F` | Applies modal filtering to suppress polynomial aliasing instabilities |
