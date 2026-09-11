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
* **Local Element Differentiation (`local_grad3`)**:
  * Implemented `Physics::mxm` matrix-matrix multiplication routine for tensor product contractions using standard loops without external dependencies.
  * Implemented `Physics::local_grad3` computing reference derivatives via tensor contractions:
    * $u_r = (\mathbf{I} \otimes \mathbf{I} \otimes \mathbf{D}) u$
    * $u_s = (\mathbf{I} \otimes \mathbf{D} \otimes \mathbf{I}) u$
    * $u_t = (\mathbf{D} \otimes \mathbf{I} \otimes \mathbf{I}) u$
* **Weighted Physical Curl (`compute_weighted_curl`)**:
  * Implemented `Physics::compute_weighted_curl` mapping local reference derivatives to weighted physical curl components matching NekCEM's `maxwell_wght_curl`.
  * Integrated curl evaluation into `Physics::computeCurl` to drive Maxwell $\mathbf{E}$ and $\mathbf{H}$ updates:
    * $\text{res}_E = \nabla \times \mathbf{H}$
    * $\text{res}_H = -\nabla \times \mathbf{E}$
* **Documentation**:
  * Created `method.md` documenting the complete mathematical and algorithmic pipeline of `cem_drive.F` and `cem_maxwell.F`.

  * Verified machine precision accuracy ($\sim 10^{-16}$) for constant and linear derivative tests and curl annihilation.
  * Verified that the entire codebase cleanly compiles under `g++ -std=c++11`.

## Missing / Next Steps
* **Mesh Topology & Multi-Element**: Implement multi-element connectivity and face mappings in `Mesh.cpp`.
* **Face Extraction**: Implement the trace mapping logic to extract boundary and interface values in `Physics::restrictToFace`.
* **Surface Numerical Flux**: Implement numerical flux formulations (e.g., Upwind or Lax-Friedrichs) in `Physics::computeFlux`.
* **Surface Flux Lifting**: Implement the lifting operator in `Physics::addFluxToRHS` to add surface flux contributions to volume residuals.
* **Mass Matrix**: Implement diagonal mass matrix inversion (GLL quadrature mass lumping) in `Physics::applyInverseMassMatrix`.
* **Boundary Conditions**: Add boundary condition handling (PEC, PMC, ABC) to the numerical flux evaluation.
* **Initial Conditions**: Set up physical initial field configurations (e.g. resonant cavity mode or Gaussian pulse) in `Orchestrator`.
