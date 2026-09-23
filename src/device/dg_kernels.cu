#include "dg_kernels.hpp"

#include <cuda_runtime.h>
#include <cstdio>

// ==============================================================================
// CUDA Device Kernels for Maxwell DG-SEM Spatial Discretization & Time Stepping
// ==============================================================================

/**
 * @brief Fused 3D tensor contraction and weighted physical curl kernel.
 *
 * Evaluates discrete curl operator for 3D Maxwell fields:
 *      dE/dt = +(1/eps) * (nabla x H)
 *      dH/dt = -(1/mu)  * (nabla x E)
 *
 * ------------------------------------------------------------------------------
 * PIOLA TRANSFORM IN VOLUME:
 * (nabla x u)_phys = (1 / J) * [ (grad_ref u) x (grad_ref x) ]
 * Reference derivatives (ur, us, ut) are converted to physical curl components
 * via metric cofactor terms of the affine coordinate transformation:
 *      (nabla x u)_x = (u3_r * ry + u3_s * sy + u3_t * ty) - (u2_r * rz + u2_s * sz + u2_t * tz)
 *      (nabla x u)_y = (u1_r * rz + u1_s * sz + u1_t * tz) - (u3_r * rx + u3_s * sx + u3_t * tx)
 *      (nabla x u)_z = (u2_r * rx + u2_s * sx + u2_t * tx) - (u1_r * ry + u1_s * sy + u1_t * ty)
 *
 * ------------------------------------------------------------------------------
 * LOCAL ELEMENT DIFFERENTIATION VIA TENSOR CONTRACTIONS (local_grad3):
 * Evaluates reference derivatives using 1D derivative matrix D:
 *      Contraction 1: ur = (I (x) I (x) D) u
 *      Contraction 2: us = (I (x) D (x) I) u
 *      Contraction 3: ut = (D (x) I (x) I) u
 *
 * ------------------------------------------------------------------------------
 * GPU OCCUPANCY & THREAD TOPOLOGY:
 *   - Grid: nelt blocks (1 block per hexahedral element)
 *   - Block: N^3 threads (1 thread per 3D GLL collocation point)
 *   - Shared Memory: Caches 1D matrix D (N^2 doubles) and element vector fields (3 * N^3 doubles)
 *   - Registers: Contraction loops unrolled in registers (#pragma unroll)
 *
 * Matches NekCEM's maxwell_wght_curl and local_grad3 routines.
 */
__global__ void gpu_volume_curl_kernel(
    const double* __restrict__ u1_all,
    const double* __restrict__ u2_all,
    const double* __restrict__ u3_all,
    double* __restrict__ w1_all,
    double* __restrict__ w2_all,
    double* __restrict__ w3_all,
    const double* __restrict__ D,
    const double* __restrict__ w3_gll,
    const double* __restrict__ jac,
    const double* __restrict__ rx, const double* __restrict__ sx, const double* __restrict__ tx,
    const double* __restrict__ ry, const double* __restrict__ sy, const double* __restrict__ ty,
    const double* __restrict__ rz, const double* __restrict__ sz, const double* __restrict__ tz,
    double sign, int N, int nelt)
{
    extern __shared__ double s_mem[];
    int N2 = N * N;
    int N3 = N * N * N;

    // Partition shared memory: 1D derivative matrix D, then vector components u1, u2, u3
    double* s_D  = s_mem;
    double* s_u1 = s_D + N2;
    double* s_u2 = s_u1 + N3;
    double* s_u3 = s_u2 + N3;

    int e = blockIdx.x;
    if (e >= nelt) return;

    int m = threadIdx.x; // Local collocation node index within element [0, N3 - 1]
    int p = e * N3 + m;  // Global collocation node index

    // Cooperatively load 1D derivative matrix D (N x N) into shared memory
    if (m < N2) {
        s_D[m] = D[m];
    }

    // Cooperatively load 3D element vector fields into shared memory
    if (m < N3) {
        s_u1[m] = u1_all[p];
        s_u2[m] = u2_all[p];
        s_u3[m] = u3_all[p];
    }

    __syncthreads();

    if (m >= N3) return;

    // Decompose 1D thread index m into 3D reference coordinates (i, j, k)
    int i = m % N;
    int j = (m / N) % N;
    int k = m / N2;

    // Accumulate reference derivatives in registers: ur, us, ut
    double u1r = 0.0, u1s = 0.0, u1t = 0.0;
    double u2r = 0.0, u2s = 0.0, u2t = 0.0;
    double u3r = 0.0, u3s = 0.0, u3t = 0.0;

    #pragma unroll
    for (int l = 0; l < N; ++l) {
        double d_il = s_D[i + l * N];
        double d_jl = s_D[j + l * N];
        double d_kl = s_D[k + l * N];

        // Contraction 1: ur = (I (x) I (x) D) u along fast index 'i'
        int idx_r = l + j * N + k * N2;
        u1r += d_il * s_u1[idx_r];
        u2r += d_il * s_u2[idx_r];
        u3r += d_il * s_u3[idx_r];

        // Contraction 2: us = (I (x) D (x) I) u across each 2D k-slice along 'j'
        int idx_s = i + l * N + k * N2;
        u1s += d_jl * s_u1[idx_s];
        u2s += d_jl * s_u2[idx_s];
        u3s += d_jl * s_u3[idx_s];

        // Contraction 3: ut = (D (x) I (x) I) u across layers along 'k'
        int idx_t = i + j * N + l * N2;
        u1t += d_kl * s_u1[idx_t];
        u2t += d_kl * s_u2[idx_t];
        u3t += d_kl * s_u3[idx_t];
    }

    // Quadrature volume weight: W_ijk * J_k
    double w = w3_gll[m] * jac[p];

    // Scale reference derivatives with quadrature metric weight
    double u1rw = u1r * w; double u1sw = u1s * w; double u1tw = u1t * w;
    double u2rw = u2r * w; double u2sw = u2s * w; double u2tw = u2t * w;
    double u3rw = u3r * w; double u3sw = u3s * w; double u3tw = u3t * w;

    // Load metric transformation factors (J^{-T})
    double rx_k = rx[p], sx_k = sx[p], tx_k = tx[p];
    double ry_k = ry[p], sy_k = sy[p], ty_k = ty[p];
    double rz_k = rz[p], sz_k = sz[p], tz_k = tz[p];

    // Physical weighted curl x-component: dw3/dy - dw2/dz
    double w1 = (u3rw * ry_k + u3sw * sy_k + u3tw * ty_k)
              - (u2rw * rz_k + u2sw * sz_k + u2tw * tz_k);

    // Physical weighted curl y-component: dw1/dz - dw3/dx
    double w2 = (u1rw * rz_k + u1sw * sz_k + u1tw * tz_k)
              - (u3rw * rx_k + u3sw * sx_k + u3tw * tx_k);

    // Physical weighted curl z-component: dw2/dx - dw1/dy
    double w3 = (u2rw * rx_k + u2sw * sx_k + u2tw * tx_k)
              - (u1rw * ry_k + u1sw * sy_k + u1tw * ty_k);

    // Store directly to global residual array (+curl(H) for E, -curl(E) for H)
    w1_all[p] = sign * w1;
    w2_all[p] = sign * w2;
    w3_all[p] = sign * w3;
}

/**
 * @brief Restricts interior volume fields to element face quadrature nodes.
 *
 * Extracts interior traces E^-, H^- onto the quadrilateral faces of each hex element.
 * Corresponds to cem_maxwell_restrict_to_face in cem_maxwell.F.
 *
 * @param state          Global state vector [Ex, Ey, Ez, Hx, Hy, Hz] (size 6 * npts)
 * @param fEN            Face trace electric field [fEx, fEy, fEz] (size 3 * totalFacePoints)
 * @param fHN            Face trace magnetic field [fHx, fHy, fHz] (size 3 * totalFacePoints)
 * @param volIdxMinus    Mapping from face point index to interior volume point index (minus side)
 * @param totalFacePoints Total number of face quadrature points across all faces
 * @param npts           Total volume collocation points (mesh.getTotalPoints())
 */
__global__ void gpu_restrict_faces_kernel(
    const double* __restrict__ state,
    double* __restrict__ fEN,
    double* __restrict__ fHN,
    const int* __restrict__ volIdxMinus,
    int totalFacePoints, int npts)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < totalFacePoints) {
        int vM = volIdxMinus[p];
        // Restrict electric field traces: E^-
        fEN[0 * totalFacePoints + p] = state[0 * npts + vM];
        fEN[1 * totalFacePoints + p] = state[1 * npts + vM];
        fEN[2 * totalFacePoints + p] = state[2 * npts + vM];

        // Restrict magnetic field traces: H^-
        fHN[0 * totalFacePoints + p] = state[3 * npts + vM];
        fHN[1 * totalFacePoints + p] = state[4 * npts + vM];
        fHN[2 * totalFacePoints + p] = state[5 * npts + vM];
    }
}

/**
 * @brief Computes upwind / central numerical flux on element interface quadrature nodes.
 *
 * Evaluates field jumps across element interfaces:
 *      [[E]] = E^+ - E^-
 *      [[H]] = H^+ - H^-
 *
 * ------------------------------------------------------------------------------
 * PIOLA TRANSFORM FOR SURFACE FLUXES:
 * Uses Nanson's formula: n * dA = J * J^{-T} * n_ref * dA_ref
 * to transform reference face normals to physical space.
 *
 * ------------------------------------------------------------------------------
 * RIEMANN / NUMERICAL FLUX SPLITTING:
 *   Upwind / Central Numerical Flux formula matching NekCEM lines 991-996:
 *      F^*_H = -0.5 * (n x [[E]]) - 0.5 * C0 * (n x (n x [[H]]))
 *      F^*_E = +0.5 * (n x [[H]]) - 0.5 * C0 * (n x (n x [[E]]))
 * where:
 *   - C0 = 0.0: Energy-conserving central flux (zero dissipation)
 *   - C0 = 1.0: Strictly dissipative upwind flux (suppresses high-frequency spurious modes)
 *
 * ------------------------------------------------------------------------------
 * PERFECT ELECTRIC CONDUCTOR (PEC) MIRROR BOUNDARY CONDITIONS:
 *   At domain boundaries with PEC conditions, mirror symmetry requires:
 *      n x E^+ = -n x E^- => [[E]] = -2 E^-
 *      n x H^+ = +n x H^- => [[H]] = 0
 * Matches cem_maxwell_flux3d and cem_maxwell_flux_pec in cem_maxwell.F.
 */
__global__ void gpu_compute_flux_kernel(
    const double* __restrict__ state,
    const double* __restrict__ fEN,
    const double* __restrict__ fHN,
    const int* __restrict__ volIdxPlus,
    const int* __restrict__ isPEC,
    const double* __restrict__ nx,
    const double* __restrict__ ny,
    const double* __restrict__ nz,
    double* __restrict__ flux,
    double c0, int totalFacePoints, int npts)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < totalFacePoints) {
        // Interior trace values (minus side)
        double Ex_m = fEN[0 * totalFacePoints + p];
        double Ey_m = fEN[1 * totalFacePoints + p];
        double Ez_m = fEN[2 * totalFacePoints + p];

        double Hx_m = fHN[0 * totalFacePoints + p];
        double Hy_m = fHN[1 * totalFacePoints + p];
        double Hz_m = fHN[2 * totalFacePoints + p];

        double dEx, dEy, dEz, dHx, dHy, dHz;
        if (isPEC[p] || volIdxPlus[p] < 0) {
            // PEC mirror conditions: [[E]] = -2 E^-, [[H]] = 0
            dEx = -2.0 * Ex_m;
            dEy = -2.0 * Ey_m;
            dEz = -2.0 * Ez_m;
            dHx = 0.0;
            dHy = 0.0;
            dHz = 0.0;
        } else {
            // Internal connection face: evaluate difference with neighbor trace (plus side)
            int vP = volIdxPlus[p];
            dEx = state[0 * npts + vP] - Ex_m;
            dEy = state[1 * npts + vP] - Ey_m;
            dEz = state[2 * npts + vP] - Ez_m;
            dHx = state[3 * npts + vP] - Hx_m;
            dHy = state[4 * npts + vP] - Hy_m;
            dHz = state[5 * npts + vP] - Hz_m;
        }

        double n_x = nx[p], n_y = ny[p], n_z = nz[p];

        // Cross product: n x [[E]]
        double nxE_x = n_y * dEz - n_z * dEy;
        double nxE_y = n_z * dEx - n_x * dEz;
        double nxE_z = n_x * dEy - n_y * dEx;

        // Cross product: n x [[H]]
        double nxH_x = n_y * dHz - n_z * dHy;
        double nxH_y = n_z * dHx - n_x * dHz;
        double nxH_z = n_x * dHy - n_y * dHx;

        // Double cross product: n x (n x [[E]])
        double nxnxE_x = n_y * nxE_z - n_z * nxE_y;
        double nxnxE_y = n_z * nxE_x - n_x * nxE_z;
        double nxnxE_z = n_x * nxE_y - n_y * nxE_x;

        // Double cross product: n x (n x [[H]])
        double nxnxH_x = n_y * nxH_z - n_z * nxH_y;
        double nxnxH_y = n_z * nxH_x - n_x * nxH_z;
        double nxnxH_z = n_x * nxH_y - n_y * nxH_x;

        // Numerical Flux formula:
        //   flxH = -0.5 * (n x [[E]]) - 0.5 * C0 * (n x (n x [[H]]))
        //   flxE = +0.5 * (n x [[H]]) - 0.5 * C0 * (n x (n x [[E]]))
        double flxHx = -0.5 * nxE_x - 0.5 * c0 * nxnxH_x;
        double flxHy = -0.5 * nxE_y - 0.5 * c0 * nxnxH_y;
        double flxHz = -0.5 * nxE_z - 0.5 * c0 * nxnxH_z;

        double flxEx = +0.5 * nxH_x - 0.5 * c0 * nxnxE_x;
        double flxEy = +0.5 * nxH_y - 0.5 * c0 * nxnxE_y;
        double flxEz = +0.5 * nxH_z - 0.5 * c0 * nxnxE_z;

        // Store into surface flux array: 0..2 for H-flux, 3..5 for E-flux
        flux[0 * totalFacePoints + p] = flxHx;
        flux[1 * totalFacePoints + p] = flxHy;
        flux[2 * totalFacePoints + p] = flxHz;
        flux[3 * totalFacePoints + p] = flxEx;
        flux[4 * totalFacePoints + p] = flxEy;
        flux[5 * totalFacePoints + p] = flxEz;
    }
}

/**
 * @brief Lifts numerical surface fluxes into volume residuals using atomicAdd.
 *
 * Adds the surface flux integrals to the volume residual via face quadrature metric areas dA:
 *      res += dA * F^*
 * Corresponds to cem_maxwell_add_flux_to_res in cem_maxwell.F.
 *
 * Uses atomicAdd because multiple faces meeting at edges/corners may contribute
 * to the same interior volume node simultaneously.
 */
__global__ void gpu_add_flux_kernel(
    const double* __restrict__ flux,
    const int* __restrict__ volIdxMinus,
    const double* __restrict__ dA,
    double* __restrict__ rhs,
    int totalFacePoints, int npts)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < totalFacePoints) {
        int vM = volIdxMinus[p];
        double a = dA[p]; // Face quadrature metric area

        // Lift electric field fluxes into E residual (resEx, resEy, resEz)
        atomicAdd(&rhs[0 * npts + vM], a * flux[3 * totalFacePoints + p]);
        atomicAdd(&rhs[1 * npts + vM], a * flux[4 * totalFacePoints + p]);
        atomicAdd(&rhs[2 * npts + vM], a * flux[5 * totalFacePoints + p]);

        // Lift magnetic field fluxes into H residual (resHx, resHy, resHz)
        atomicAdd(&rhs[3 * npts + vM], a * flux[0 * totalFacePoints + p]);
        atomicAdd(&rhs[4 * npts + vM], a * flux[1 * totalFacePoints + p]);
        atomicAdd(&rhs[5 * npts + vM], a * flux[2 * totalFacePoints + p]);
    }
}

/**
 * @brief Multiplies residuals by diagonal inverse mass matrix (GLL quadrature mass lumping).
 *
 * Exact diagonal mass matrix inversion:
 *      res_E = res_E / (eps * J * w_i * w_j * w_k)
 *      res_H = res_H / (mu  * J * w_i * w_j * w_k)
 *
 * Due to Gauss-Lobatto-Legendre (GLL) quadrature, the mass matrix is strictly diagonal
 * without any loss of high-order spectral accuracy (mass lumping property).
 * Vacuum material parameters: eps = 1.0, mu = 1.0.
 *
 * Corresponds to cem_maxwell_invqmass in cem_maxwell.F.
 */
__global__ void gpu_inv_mass_kernel(
    double* __restrict__ rhs,
    const double* __restrict__ jac,
    const double* __restrict__ w3,
    int nelt, int nxyz, int npts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nelt * nxyz) {
        int e = idx / nxyz;
        int i = idx % nxyz;
        int k = e * nxyz + i;

        // Material parameters (vacuum: eps = 1.0, mu = 1.0)
        // We use here normalized quantities!
        const double eps = 1.0;
        const double mu = 1.0;
        double invMass_E = 1.0 / (eps * jac[k] * w3[i]);
        double invMass_H = 1.0 / (mu * jac[k] * w3[i]);

        // Scale electric field residual components: resEx, resEy, resEz
        rhs[0 * npts + k] *= invMass_E;
        rhs[1 * npts + k] *= invMass_E;
        rhs[2 * npts + k] *= invMass_E;

        // Scale magnetic field residual components: resHx, resHy, resHz
        rhs[3 * npts + k] *= invMass_H;
        rhs[4 * npts + k] *= invMass_H;
        rhs[5 * npts + k] *= invMass_H;
    }
}

/**
 * @brief Vectorized Low-Storage Runge-Kutta 4th-order 5-stage state update.
 *
 * Implements Carpenter & Kennedy (1994) low-storage update formula (matching NekCEM's rk4_upd):
 *      k = a_s * k + dt * rhs
 *      state = state + b_s * k
 *
 * Operates across all 6 electromagnetic fields simultaneously (totalEntries = 6 * npts).
 */
__global__ void gpu_lsrk45_update_kernel(
    double* __restrict__ state,
    double* __restrict__ k,
    const double* __restrict__ rhs,
    double a_s, double b_s, double dt,
    int totalEntries)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalEntries) {
        double ki = a_s * k[i] + dt * rhs[i];
        k[i] = ki;
        state[i] += b_s * ki;
    }
}

// ==============================================================================
// Launcher Dispatch Wrappers (Exposed C++ API)
// ==============================================================================

namespace nekwave {
namespace device {

/**
 * @brief Dispatches volume curl kernel with 1 block per element and N^3 threads per block.
 *
 * Allocates dynamic shared memory for 1D derivative matrix D (N^2 doubles)
 * and 3D element vector fields u1, u2, u3 (3 * N^3 doubles).
 */
void launch_volume_curl(
    const double* u1, const double* u2, const double* u3,
    double* w1, double* w2, double* w3,
    const double* D, const double* w3_gll, const double* jac,
    const double* rx, const double* sx, const double* tx,
    const double* ry, const double* sy, const double* ty,
    const double* rz, const double* sz, const double* tz,
    double sign, int N, int nelt, cudaStream_t stream)
{
    int nxyz = N * N * N;
    int blockSize = nxyz;
    int numBlocks = nelt;
    size_t shmemBytes = (N * N + 3 * nxyz) * sizeof(double);

    gpu_volume_curl_kernel<<<numBlocks, blockSize, shmemBytes, stream>>>(
        u1, u2, u3, w1, w2, w3,
        D, w3_gll, jac,
        rx, sx, tx,
        ry, sy, ty,
        rz, sz, tz,
        sign, N, nelt
    );
    NW_GPU_CHECK(cudaGetLastError());
}

/**
 * @brief Dispatches face restriction kernel to extract interior field traces.
 */
void launch_restrict_faces(
    const double* d_state, double* d_fEN, double* d_fHN,
    const int* d_volIdxMinus, int totalFacePoints, int npts,
    cudaStream_t stream)
{
    int blockSize = 256;
    int numBlocks = (totalFacePoints + blockSize - 1) / blockSize;

    gpu_restrict_faces_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_state, d_fEN, d_fHN, d_volIdxMinus, totalFacePoints, npts
    );
    NW_GPU_CHECK(cudaGetLastError());
}

/**
 * @brief Dispatches numerical surface flux kernel (Central or Upwind + PEC mirror).
 */
void launch_compute_flux(
    const double* d_state, const double* d_fEN, const double* d_fHN,
    const int* d_volIdxPlus, const int* d_isPEC,
    const double* d_nx, const double* d_ny, const double* d_nz,
    double* d_flux, double c0, int totalFacePoints, int npts,
    cudaStream_t stream)
{
    int blockSize = 256;
    int numBlocks = (totalFacePoints + blockSize - 1) / blockSize;

    gpu_compute_flux_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_state, d_fEN, d_fHN, d_volIdxPlus, d_isPEC,
        d_nx, d_ny, d_nz, d_flux, c0, totalFacePoints, npts
    );
    NW_GPU_CHECK(cudaGetLastError());
}

/**
 * @brief Dispatches surface flux lifting kernel into volume residuals using atomic additions.
 */
void launch_add_flux(
    const double* d_flux, const int* d_volIdxMinus, const double* d_dA,
    double* d_rhs, int totalFacePoints, int npts,
    cudaStream_t stream)
{
    int blockSize = 256;
    int numBlocks = (totalFacePoints + blockSize - 1) / blockSize;

    gpu_add_flux_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_flux, d_volIdxMinus, d_dA, d_rhs, totalFacePoints, npts
    );
    NW_GPU_CHECK(cudaGetLastError());
}

/**
 * @brief Dispatches diagonal inverse mass matrix scaling kernel across all volume points.
 */
void launch_inv_mass(
    double* d_rhs, const double* d_jac, const double* d_w3,
    int nelt, int nxyz, int npts,
    cudaStream_t stream)
{
    int blockSize = 256;
    int numBlocks = (nelt * nxyz + blockSize - 1) / blockSize;

    gpu_inv_mass_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_rhs, d_jac, d_w3, nelt, nxyz, npts
    );
    NW_GPU_CHECK(cudaGetLastError());
}

/**
 * @brief Dispatches vectorized 5-stage Low-Storage Runge-Kutta state update kernel.
 */
void launch_lsrk45_update(
    double* d_state, double* d_k, const double* d_rhs,
    double rk4a, double rk4b, double dt, int totalEntries,
    cudaStream_t stream)
{
    int blockSize = 256;
    int numBlocks = (totalEntries + blockSize - 1) / blockSize;

    gpu_lsrk45_update_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_state, d_k, d_rhs, rk4a, rk4b, dt, totalEntries
    );
    NW_GPU_CHECK(cudaGetLastError());
}

} // namespace device
} // namespace nekwave
