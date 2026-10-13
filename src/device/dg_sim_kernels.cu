#include "dg_sim_kernels.hpp"
#include "gpu_runtime.hpp"
#include "boundary_conditions.hpp"
#include <cstdio>

// ==============================================================================
// CUDA/HIP Device Kernels for Maxwell DG-SEM Spatial Discretization & Time Stepping
// ==============================================================================

/**
 * @brief Fused Dual-Field 3D Tensor Contraction and Weighted Physical Curl Kernel.
 *
 * Performs TWO coupled volume curl operations in a single GPU kernel pass:
 *   - OPERATION 1 (+curl H -> resE): Evaluates discrete reference curl of magnetic field H
 *                                    for Faraday/Ampere-Maxwell electric residual:
 *                                        res_E = +(W * J) * (nabla x H)_phys
 *   - OPERATION 2 (-curl E -> resH): Evaluates discrete reference curl of electric field E
 *                                    for Faraday magnetic residual:
 *                                        res_H = -(W * J) * (nabla x E)_phys
 *
 * ------------------------------------------------------------------------------
 * WHY THESE TWO OPERATIONS ARE FUSED:
 *   1. Both curls share the exact same 1D spectral derivative matrix D (cached in LDS),
 *      the same 3D GLL quadrature weights w3_gll, the same Jacobian J, and the same 9
 *      geometric metric cofactor arrays (rx, sx, tx, ry, sy, ty, rz, sz, tz).
 *   2. Fusing both curls eliminates an entire HBM pass over the 10 geometric metric arrays
 *      (80 Bytes/point saved per stage = 400 Bytes/point saved per 5-stage LSRK45 step),
 *      increasing the volume kernel Arithmetic Intensity from ~1.55 to ~2.26 FLOP/Byte.
 *   3. Shared memory (LDS) is reused across Operation 1 (H) and Operation 2 (E) so the
 *      per-block LDS footprint remains (N^2 + 3 * N^3) doubles, preserving GPU occupancy.
 *
 * ------------------------------------------------------------------------------
 * PIOLA TRANSFORM IN VOLUME:
 *   (nabla x u)_phys = (1 / J) * [ (grad_ref u) x (grad_ref x) ]
 *   Reference derivatives (ur, us, ut) are converted to physical curl components:
 *     (nabla x u)_x = (u3_r * ry + u3_s * sy + u3_t * ty) - (u2_r * rz + u2_s * sz + u2_t * tz)
 *     (nabla x u)_y = (u1_r * rz + u1_s * sz + u1_t * tz) - (u3_r * rx + u3_s * sx + u3_t * tx)
 *     (nabla x u)_z = (u2_r * rx + u2_s * sx + u2_t * tx) - (u1_r * ry + u1_s * sy + u1_t * ty)
 *
 * Matches NekCEM's maxwell_wght_curl and local_grad3 routines for both H and E.
 */
__global__ void gpu_volume_curl_kernel(
    const double* __restrict__ Ex_all,
    const double* __restrict__ Ey_all,
    const double* __restrict__ Ez_all,
    const double* __restrict__ Hx_all,
    const double* __restrict__ Hy_all,
    const double* __restrict__ Hz_all,
    double* __restrict__ resEx_all,
    double* __restrict__ resEy_all,
    double* __restrict__ resEz_all,
    double* __restrict__ resHx_all,
    double* __restrict__ resHy_all,
    double* __restrict__ resHz_all,
    const double* __restrict__ D,
    const double* __restrict__ w3_gll,
    const double* __restrict__ jac,
    const double* __restrict__ rx, const double* __restrict__ sx, const double* __restrict__ tx,
    const double* __restrict__ ry, const double* __restrict__ sy, const double* __restrict__ ty,
    const double* __restrict__ rz, const double* __restrict__ sz, const double* __restrict__ tz,
    int N, int nelt)
{
    extern __shared__ double s_mem[];
    int N2 = N * N;
    int N3 = N2 * N;

    // Partition shared memory: 1D derivative matrix D (N^2), plus 3D element vector buffer (3 * N^3)
    double* s_D  = s_mem;
    double* s_u1 = s_D + N2;
    double* s_u2 = s_u1 + N3;
    double* s_u3 = s_u2 + N3;

    int e = blockIdx.x;
    if (e >= nelt) return;

    int m = threadIdx.x; // Local collocation node index within element [0, N3 - 1]
    int p = e * N3 + m;  // Global collocation node index

    // Cooperatively load 1D spectral derivative matrix D (N x N) into shared memory
    if (m < N2) {
        s_D[m] = D[m];
    }

    // Decompose 1D thread index m into 3D reference coordinates (i, j, k)
    int i = m % N;
    int j = (m / N) % N;
    int k = m / N2;

    // ==========================================================================
    // OPERATION 1 (Part A): Reference Tensor Contraction of Magnetic Field H
    //   Evaluates H_r = (I x I x D) H,  H_s = (I x D x I) H,  H_t = (D x I x I) H
    // ==========================================================================
    if (m < N3) {
        s_u1[m] = Hx_all[p];
        s_u2[m] = Hy_all[p];
        s_u3[m] = Hz_all[p];
    }
    __syncthreads();

    double Hx_r = 0.0, Hx_s = 0.0, Hx_t = 0.0;
    double Hy_r = 0.0, Hy_s = 0.0, Hy_t = 0.0;
    double Hz_r = 0.0, Hz_s = 0.0, Hz_t = 0.0;

    if (m < N3) {
        #pragma unroll
        for (int l = 0; l < N; ++l) {
            double d_il = s_D[i + l * N];
            double d_jl = s_D[j + l * N];
            double d_kl = s_D[k + l * N];

            int idx_r = l + j * N + k * N2;
            Hx_r += d_il * s_u1[idx_r];
            Hy_r += d_il * s_u2[idx_r];
            Hz_r += d_il * s_u3[idx_r];

            int idx_s = i + l * N + k * N2;
            Hx_s += d_jl * s_u1[idx_s];
            Hy_s += d_jl * s_u2[idx_s];
            Hz_s += d_jl * s_u3[idx_s];

            int idx_t = i + j * N + l * N2;
            Hx_t += d_kl * s_u1[idx_t];
            Hy_t += d_kl * s_u2[idx_t];
            Hz_t += d_kl * s_u3[idx_t];
        }
    }
    __syncthreads();

    // ==========================================================================
    // OPERATION 2 (Part A): Reference Tensor Contraction of Electric Field E
    //   Reuses shared memory s_u1..s_u3 to evaluate E_r, E_s, E_t
    // ==========================================================================
    if (m < N3) {
        s_u1[m] = Ex_all[p];
        s_u2[m] = Ey_all[p];
        s_u3[m] = Ez_all[p];
    }
    __syncthreads();

    if (m >= N3) return;

    double Ex_r = 0.0, Ex_s = 0.0, Ex_t = 0.0;
    double Ey_r = 0.0, Ey_s = 0.0, Ey_t = 0.0;
    double Ez_r = 0.0, Ez_s = 0.0, Ez_t = 0.0;

    #pragma unroll
    for (int l = 0; l < N; ++l) {
        double d_il = s_D[i + l * N];
        double d_jl = s_D[j + l * N];
        double d_kl = s_D[k + l * N];

        int idx_r = l + j * N + k * N2;
        Ex_r += d_il * s_u1[idx_r];
        Ey_r += d_il * s_u2[idx_r];
        Ez_r += d_il * s_u3[idx_r];

        int idx_s = i + l * N + k * N2;
        Ex_s += d_jl * s_u1[idx_s];
        Ey_s += d_jl * s_u2[idx_s];
        Ez_s += d_jl * s_u3[idx_s];

        int idx_t = i + j * N + l * N2;
        Ex_t += d_kl * s_u1[idx_t];
        Ey_t += d_kl * s_u2[idx_t];
        Ez_t += d_kl * s_u3[idx_t];
    }

    // ==========================================================================
    // SHARED METRIC LOAD & PHYSICAL PIOLA CURL TRANSFORMATION (Operations 1 & 2)
    //   Loads Jacobian J and 9 contravariant metric factors (rx..tz) ONCE from HBM
    //   and applies them to both +curl(H) and -curl(E).
    // ==========================================================================
    double w = w3_gll[m] * jac[p];

    double rx_k = rx[p], sx_k = sx[p], tx_k = tx[p];
    double ry_k = ry[p], sy_k = sy[p], ty_k = ty[p];
    double rz_k = rz[p], sz_k = sz[p], tz_k = tz[p];

    // --- Operation 1 (Part B): Physical weighted curl of H -> +curl(H) into resE ---
    double Hx_rw = Hx_r * w; double Hx_sw = Hx_s * w; double Hx_tw = Hx_t * w;
    double Hy_rw = Hy_r * w; double Hy_sw = Hy_s * w; double Hy_tw = Hy_t * w;
    double Hz_rw = Hz_r * w; double Hz_sw = Hz_s * w; double Hz_tw = Hz_t * w;

    double curlH_x = (Hz_rw * ry_k + Hz_sw * sy_k + Hz_tw * ty_k)
                   - (Hy_rw * rz_k + Hy_sw * sz_k + Hy_tw * tz_k);
    double curlH_y = (Hx_rw * rz_k + Hx_sw * sz_k + Hx_tw * tz_k)
                   - (Hz_rw * rx_k + Hz_sw * sx_k + Hz_tw * tx_k);
    double curlH_z = (Hy_rw * rx_k + Hy_sw * sx_k + Hy_tw * tx_k)
                   - (Hx_rw * ry_k + Hx_sw * sy_k + Hx_tw * ty_k);

    resEx_all[p] =  curlH_x;
    resEy_all[p] =  curlH_y;
    resEz_all[p] =  curlH_z;

    // --- Operation 2 (Part B): Physical weighted curl of E -> -curl(E) into resH ---
    double Ex_rw = Ex_r * w; double Ex_sw = Ex_s * w; double Ex_tw = Ex_t * w;
    double Ey_rw = Ey_r * w; double Ey_sw = Ey_s * w; double Ey_tw = Ey_t * w;
    double Ez_rw = Ez_r * w; double Ez_sw = Ez_s * w; double Ez_tw = Ez_t * w;

    double curlE_x = (Ez_rw * ry_k + Ez_sw * sy_k + Ez_tw * ty_k)
                   - (Ey_rw * rz_k + Ey_sw * sz_k + Ey_tw * tz_k);
    double curlE_y = (Ex_rw * rz_k + Ex_sw * sz_k + Ex_tw * tz_k)
                   - (Ez_rw * rx_k + Ez_sw * sx_k + Ez_tw * tx_k);
    double curlE_z = (Ey_rw * rx_k + Ey_sw * sx_k + Ey_tw * tx_k)
                   - (Ex_rw * ry_k + Ex_sw * sy_k + Ex_tw * ty_k);

    resHx_all[p] = -curlE_x;
    resHy_all[p] = -curlE_y;
    resHz_all[p] = -curlE_z;
}

/**
 * @brief Evaluates field jumps [[E]] = E^+ - E^- and [[H]] = H^+ - H^- across a face point
 *        according to its BcType (INTERIOR, PERIODIC, MPI_CUT, PEC, PMC, PML).
 */
__device__ __forceinline__ void apply_boundary_condition_jump(
    int p, int bcCode, int vP,
    const double* __restrict__ state, int stateStride,
    double Ex_m, double Ey_m, double Ez_m,
    double Hx_m, double Hy_m, double Hz_m,
    const double* __restrict__ fx,
    const double* __restrict__ fy,
    const double* __restrict__ fz,
    const IncidentPlaneWaveConfig& incWave,
    double stageTime,
    double& dEx, double& dEy, double& dEz,
    double& dHx, double& dHy, double& dHz)
{
    const BcType bc = static_cast<BcType>(bcCode);

    if (bc == BcType::PMC) {
        // Perfect Magnetic Conductor (PMC / Symmetry):
        //   n x E^+ = +n x E^- => [[E]] = 0
        //   n x H^+ = -n x H^- => [[H]] = -2 H^-
        dEx = 0.0;
        dEy = 0.0;
        dEz = 0.0;
        dHx = -2.0 * Hx_m;
        dHy = -2.0 * Hy_m;
        dHz = -2.0 * Hz_m;
    } else if (bc == BcType::PEC) {
        // Perfect Electric Conductor (PEC) scatterer boundary:
        //   Total-field PEC:     n x E^+ = -n x E^-                 => [[E]] = -2 E^-
        //   Scattered-field PEC: n x (E_scat + E_inc) = 0 on wall   => [[E_scat]] = -2 (E_scat^- + E_inc)
        //   and [[H]] = 0 in both formulations.
        if (incWave.enabled && fx != nullptr && fy != nullptr && fz != nullptr) {
            double Ex_inc = 0.0, Ey_inc = 0.0, Ez_inc = 0.0;
            double Hx_inc = 0.0, Hy_inc = 0.0, Hz_inc = 0.0;
            incWave.evaluate(fx[p], fy[p], fz[p], stageTime,
                             Ex_inc, Ey_inc, Ez_inc,
                             Hx_inc, Hy_inc, Hz_inc);
            dEx = -2.0 * (Ex_m + Ex_inc);
            dEy = -2.0 * (Ey_m + Ey_inc);
            dEz = -2.0 * (Ez_m + Ez_inc);
        } else {
            dEx = -2.0 * Ex_m;
            dEy = -2.0 * Ey_m;
            dEz = -2.0 * Ez_m;
        }
        dHx = 0.0;
        dHy = 0.0;
        dHz = 0.0;
    } else if (bc == BcType::PML || vP < 0) {
        // PML outer backing wall (PEC mirror termination for absorbed scattered/total wave):
        dEx = -2.0 * Ex_m;
        dEy = -2.0 * Ey_m;
        dEz = -2.0 * Ez_m;
        dHx = 0.0;
        dHy = 0.0;
        dHz = 0.0;
    } else {
        // Internal conforming face (INTERIOR), Periodic wrap-around (PERIODIC), or MPI halo (MPI_CUT):
        // evaluate difference with neighbor trace (plus side) at vP
        dEx = state[0 * stateStride + vP] - Ex_m;
        dEy = state[1 * stateStride + vP] - Ey_m;
        dEz = state[2 * stateStride + vP] - Ez_m;
        dHx = state[3 * stateStride + vP] - Hx_m;
        dHy = state[4 * stateStride + vP] - Hy_m;
        dHz = state[5 * stateStride + vP] - Hz_m;
    }
}

/**
 * @brief Fused 3-Stage Surface Interface Kernel:
 *        (1) Face Trace Restriction  +  (2) Upwind/Central Riemann Flux  +  (3) Surface Flux Lift
 *
 * Combines three formerly separate surface operations into a single register-resident pass
 * over all surface quadrature nodes p in [0, totalFacePoints):
 *
 *   - OPERATION 1 (Face Trace Restriction — NekCEM cem_maxwell_restrict_to_face):
 *       Extracts interior volume trace fields (E^-, H^-) at vM = volIdxMinus[p]
 *       directly into registers without writing intermediate trace arrays (fEN, fHN) to HBM.
 *
 *   - OPERATION 2 (Riemann Numerical Flux — NekCEM cem_maxwell_flux3d & cem_maxwell_flux_pec):
 *       Evaluates interface field jumps [[E]] = E^+ - E^- and [[H]] = H^+ - H^- across
 *       interior, periodic, MPI-cut, PEC, PMC, and PML boundaries, and computes the
 *       Upwind/Central numerical flux in registers:
 *           F^*_H = -0.5 * (n x [[E]]) - 0.5 * C0 * (n x (n x [[H]]))
 *           F^*_E = +0.5 * (n x [[H]]) - 0.5 * C0 * (n x (n x [[E]]))
 *       without writing an intermediate flux array (d_flux) to HBM.
 *
 *   - OPERATION 3 (Surface Flux Lifting — NekCEM cem_maxwell_add_flux_to_res):
 *       Integrates the numerical flux against the face quadrature metric area a = dA[p]
 *       and lifts it into the volume residual:
 *           rhs[vM] += dA[p] * F^*
 *       For strictly interior face GLL points (0 < pA < N-1 and 0 < qA < N-1), vM is
 *       uniquely owned by a single face point p, so a direct non-atomic update is used;
 *       element edge and corner nodes shared by 2 or 3 faces of the same element use atomicAdd.
 */
__global__ void gpu_surface_flux_kernel(
    const double* __restrict__ state,
    const int* __restrict__ volIdxMinus,
    const int* __restrict__ volIdxPlus,
    const int* __restrict__ bcType,
    const double* __restrict__ nx,
    const double* __restrict__ ny,
    const double* __restrict__ nz,
    const double* __restrict__ dA,
    double* __restrict__ rhs,
    double c0, int N, int totalFacePoints, int npts, int stateStride,
    const double* __restrict__ fx,
    const double* __restrict__ fy,
    const double* __restrict__ fz,
    IncidentPlaneWaveConfig incWave,
    double stageTime)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= totalFacePoints) return;

    // ==========================================================================
    // OPERATION 1: Restrict Interior Volume Solution Fields to Face Trace (E^-, H^-)
    // ==========================================================================
    const int vM = volIdxMinus[p];
    const double Ex_m = state[0 * stateStride + vM];
    const double Ey_m = state[1 * stateStride + vM];
    const double Ez_m = state[2 * stateStride + vM];

    const double Hx_m = state[3 * stateStride + vM];
    const double Hy_m = state[4 * stateStride + vM];
    const double Hz_m = state[5 * stateStride + vM];

    // ==========================================================================
    // OPERATION 2: Evaluate Interface Jumps ([[E]], [[H]]) & Upwind/Central Riemann Flux
    // ==========================================================================
    double dEx, dEy, dEz, dHx, dHy, dHz;
    apply_boundary_condition_jump(
        p, bcType[p], volIdxPlus[p], state, stateStride,
        Ex_m, Ey_m, Ez_m, Hx_m, Hy_m, Hz_m,
        fx, fy, fz, incWave, stageTime,
        dEx, dEy, dEz, dHx, dHy, dHz
    );

    const double n_x = nx[p], n_y = ny[p], n_z = nz[p];

    // Cross product: n x [[E]]
    const double nxE_x = n_y * dEz - n_z * dEy;
    const double nxE_y = n_z * dEx - n_x * dEz;
    const double nxE_z = n_x * dEy - n_y * dEx;

    // Cross product: n x [[H]]
    const double nxH_x = n_y * dHz - n_z * dHy;
    const double nxH_y = n_z * dHx - n_x * dHz;
    const double nxH_z = n_x * dHy - n_y * dHx;

    // Double cross product: n x (n x [[E]])
    const double nxnxE_x = n_y * nxE_z - n_z * nxE_y;
    const double nxnxE_y = n_z * nxE_x - n_x * nxE_z;
    const double nxnxE_z = n_x * nxE_y - n_y * nxE_x;

    // Double cross product: n x (n x [[H]])
    const double nxnxH_x = n_y * nxH_z - n_z * nxH_y;
    const double nxnxH_y = n_z * nxH_x - n_x * nxH_z;
    const double nxnxH_z = n_x * nxH_y - n_y * nxH_x;

    // Numerical Flux formula:
    //   flxH = -0.5 * (n x [[E]]) - 0.5 * C0 * (n x (n x [[H]]))
    //   flxE = +0.5 * (n x [[H]]) - 0.5 * C0 * (n x (n x [[E]]))
    const double flxHx = -0.5 * nxE_x - 0.5 * c0 * nxnxH_x;
    const double flxHy = -0.5 * nxE_y - 0.5 * c0 * nxnxH_y;
    const double flxHz = -0.5 * nxE_z - 0.5 * c0 * nxnxH_z;

    const double flxEx = +0.5 * nxH_x - 0.5 * c0 * nxnxE_x;
    const double flxEy = +0.5 * nxH_y - 0.5 * c0 * nxnxE_y;
    const double flxEz = +0.5 * nxH_z - 0.5 * c0 * nxnxE_z;

    // ==========================================================================
    // OPERATION 3: Lift / Scatter Numerical Surface Flux into Volume Residual (rhs += dA * F^*)
    // ==========================================================================
    const double a = dA[p];
    const int N2 = N * N;
    const int localPt = p % N2;
    const int pA = localPt % N;
    const int qA = localPt / N;

    // Interior face GLL nodes (0 < pA < N-1 and 0 < qA < N-1) have exactly one
    // coordinate on the element boundary and are never shared with any other face,
    // allowing direct non-atomic accumulation into rhs. Edge/corner nodes use atomicAdd.
    if (pA > 0 && pA < N - 1 && qA > 0 && qA < N - 1) {
        rhs[0 * npts + vM] += a * flxEx;
        rhs[1 * npts + vM] += a * flxEy;
        rhs[2 * npts + vM] += a * flxEz;
        rhs[3 * npts + vM] += a * flxHx;
        rhs[4 * npts + vM] += a * flxHy;
        rhs[5 * npts + vM] += a * flxHz;
    } else {
        atomicAdd(&rhs[0 * npts + vM], a * flxEx);
        atomicAdd(&rhs[1 * npts + vM], a * flxEy);
        atomicAdd(&rhs[2 * npts + vM], a * flxEz);
        atomicAdd(&rhs[3 * npts + vM], a * flxHx);
        atomicAdd(&rhs[4 * npts + vM], a * flxHy);
        atomicAdd(&rhs[5 * npts + vM], a * flxHz);
    }
}

/**
 * @brief Fused 2-Stage UPML Absorbing Boundary Kernel:
 *        (1) UPML Auxiliary Differential Equations (ADE)  +  (2) Auxiliary Inverse Mass & LSRK45 Update
 *
 * Combines two UPML operations into a single pass over all PML collocation nodes:
 *
 *   - OPERATION 1 (UPML ADE Coupling — NekCEM pml_step in cem_maxwell_pml.F):
 *       Couples the unscaled spatial curl + flux residual (resPr for E, resQr for H)
 *       with the uniaxial conductivity profile (sx, sy, sz) and auxiliary fields (DN, BN):
 *           resDN_x = resPr_x - bm1 * sy * DN_x,  ...
 *           resEN_x = resDN_x + bm1 * (sx - sz) * DN_x - bm1 * sz * EN_x,  ...
 *       and writes the updated unscaled primary residual back to rhs[k].
 *
 *   - OPERATION 2 (Auxiliary Inverse Mass Scaling & LSRK45 Stage Update):
 *       Scales the auxiliary residuals (resDN, resBN) by 1 / bm1 = 1 / (J * w_ijk) in registers
 *       and immediately advances the auxiliary LSRK45 register kPmlAux and state pmlAux:
 *           kPmlAux = a_s * kPmlAux + dt * (resAux / bm1)
 *           pmlAux  = pmlAux + b_s * kPmlAux
 *       eliminating the intermediate d_resPmlAux global memory buffer and second LSRK45 launch.
 */
__global__ void gpu_pml_step_kernel(
    const double* __restrict__ state,
    double* __restrict__ rhs,
    double* __restrict__ pmlAux,
    double* __restrict__ kPmlAux,
    const int* __restrict__ pmlPtr,
    const double* __restrict__ pmlSigma,
    const double* __restrict__ jac,
    const double* __restrict__ w3,
    double a_s, double b_s, double dt,
    int maxPml, int nxyz, int npts, int stateStride)
{
    int pmlIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nptsPml = maxPml * nxyz;
    if (pmlIdx < nptsPml) {
        int ie = pmlIdx / nxyz;
        int i  = pmlIdx % nxyz;
        int e  = pmlPtr[ie];
        int k  = e * nxyz + i;

        double bm1    = jac[k] * w3[i];
        double invBm1 = 1.0 / bm1;

        double sx = pmlSigma[0 * npts + k];
        double sy = pmlSigma[1 * npts + k];
        double sz = pmlSigma[2 * npts + k];

        // Normalized vacuum permittivity and permeability (eps = 1.0, mu = 1.0)
        const double inv_eps = 1.0;
        const double inv_mu  = 1.0;

        // ======================================================================
        // OPERATION 1: UPML Auxiliary Differential Equations (ADE) & RHS Coupling
        // ======================================================================
        double resPr1 = rhs[0 * npts + k];
        double resPr2 = rhs[1 * npts + k];
        double resPr3 = rhs[2 * npts + k];
        double resQr1 = rhs[3 * npts + k];
        double resQr2 = rhs[4 * npts + k];
        double resQr3 = rhs[5 * npts + k];

        double Dn1 = pmlAux[0 * nptsPml + pmlIdx];
        double Dn2 = pmlAux[1 * nptsPml + pmlIdx];
        double Dn3 = pmlAux[2 * nptsPml + pmlIdx];
        double Bn1 = pmlAux[3 * nptsPml + pmlIdx];
        double Bn2 = pmlAux[4 * nptsPml + pmlIdx];
        double Bn3 = pmlAux[5 * nptsPml + pmlIdx];

        double Ex = state[0 * stateStride + k];
        double Ey = state[1 * stateStride + k];
        double Ez = state[2 * stateStride + k];
        double Hx = state[3 * stateStride + k];
        double Hy = state[4 * stateStride + k];
        double Hz = state[5 * stateStride + k];

        double resDn1 = resPr1 - bm1 * sy * inv_eps * Dn1;
        double resDn2 = resPr2 - bm1 * sz * inv_eps * Dn2;
        double resDn3 = resPr3 - bm1 * sx * inv_eps * Dn3;

        double resBn1 = resQr1 - bm1 * sy * inv_eps * Bn1;
        double resBn2 = resQr2 - bm1 * sz * inv_eps * Bn2;
        double resBn3 = resQr3 - bm1 * sx * inv_eps * Bn3;

        rhs[0 * npts + k] = resDn1 + bm1 * sx * inv_eps * Dn1 - bm1 * sz * Ex;
        rhs[1 * npts + k] = resDn2 + bm1 * sy * inv_eps * Dn2 - bm1 * sx * Ey;
        rhs[2 * npts + k] = resDn3 + bm1 * sz * inv_eps * Dn3 - bm1 * sy * Ez;

        rhs[3 * npts + k] = resBn1 + bm1 * sx * inv_mu  * Bn1 - bm1 * sz * Hx;
        rhs[4 * npts + k] = resBn2 + bm1 * sy * inv_mu  * Bn2 - bm1 * sx * Hy;
        rhs[5 * npts + k] = resBn3 + bm1 * sz * inv_mu  * Bn3 - bm1 * sy * Hz;

        // ======================================================================
        // OPERATION 2: Auxiliary Inverse Mass Scaling & LSRK45 Stage Update
        //   Scales resDn, resBn by invBm1 = 1 / (J * w_ijk) in registers and
        //   updates kPmlAux and pmlAux in-place.
        // ======================================================================
        double kDn1 = a_s * kPmlAux[0 * nptsPml + pmlIdx] + dt * (resDn1 * invBm1);
        double kDn2 = a_s * kPmlAux[1 * nptsPml + pmlIdx] + dt * (resDn2 * invBm1);
        double kDn3 = a_s * kPmlAux[2 * nptsPml + pmlIdx] + dt * (resDn3 * invBm1);
        double kBn1 = a_s * kPmlAux[3 * nptsPml + pmlIdx] + dt * (resBn1 * invBm1);
        double kBn2 = a_s * kPmlAux[4 * nptsPml + pmlIdx] + dt * (resBn2 * invBm1);
        double kBn3 = a_s * kPmlAux[5 * nptsPml + pmlIdx] + dt * (resBn3 * invBm1);

        kPmlAux[0 * nptsPml + pmlIdx] = kDn1;
        kPmlAux[1 * nptsPml + pmlIdx] = kDn2;
        kPmlAux[2 * nptsPml + pmlIdx] = kDn3;
        kPmlAux[3 * nptsPml + pmlIdx] = kBn1;
        kPmlAux[4 * nptsPml + pmlIdx] = kBn2;
        kPmlAux[5 * nptsPml + pmlIdx] = kBn3;

        pmlAux[0 * nptsPml + pmlIdx] = Dn1 + b_s * kDn1;
        pmlAux[1 * nptsPml + pmlIdx] = Dn2 + b_s * kDn2;
        pmlAux[2 * nptsPml + pmlIdx] = Dn3 + b_s * kDn3;
        pmlAux[3 * nptsPml + pmlIdx] = Bn1 + b_s * kBn1;
        pmlAux[4 * nptsPml + pmlIdx] = Bn2 + b_s * kBn2;
        pmlAux[5 * nptsPml + pmlIdx] = Bn3 + b_s * kBn3;
    }
}

/**
 * @brief Fused 2-Stage Diagonal Inverse Mass Scaling and LSRK45 Stage Update Kernel:
 *        (1) Diagonal Inverse Mass Matrix Scaling  +  (2) Low-Storage Runge-Kutta 4(5) Update
 *
 * Combines the final two volume operations of each LSRK45 stage into a single pass
 * over all local volume GLL collocation nodes p in [0, npts):
 *
 *   - OPERATION 1 (Diagonal Inverse Mass Matrix Scaling — NekCEM cem_maxwell_invqmass):
 *       Exploits Gauss-Lobatto-Legendre (GLL) quadrature mass lumping to invert the
 *       diagonal mass matrix in registers:
 *           res_scaled = rhs[p] / (J[p] * w_i * w_j * w_k)
 *       without writing the scaled residual back to global HBM memory.
 *
 *   - OPERATION 2 (LSRK45 State & Auxiliary Register Update — NekCEM rk4_upd):
 *       Immediately applies Carpenter & Kennedy's (1994) 5-stage 4th-order low-storage
 *       Runge-Kutta update across all 6 electromagnetic field components (Ex..Hz):
 *           k[p]     = a_s * k[p] + dt * res_scaled
 *           state[p] = state[p]   + b_s * k[p]
 */
__global__ void gpu_inv_mass_lsrk45_update_kernel(
    double* __restrict__ state,
    double* __restrict__ k,
    const double* __restrict__ rhs,
    const double* __restrict__ jac,
    const double* __restrict__ w3,
    double a_s, double b_s, double dt,
    int nelt, int nxyz, int npts, int stateStride)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < nelt * nxyz) {
        int i = p % nxyz;

        // ======================================================================
        // OPERATION 1: Diagonal Inverse Mass Matrix Scaling (GLL Mass Lumping)
        //   invMass_E = 1.0 / (eps * J[p] * w3[i]),  invMass_H = 1.0 / (mu * J[p] * w3[i])
        //   (Normalized vacuum parameters: eps = 1.0, mu = 1.0)
        // ======================================================================
        const double invMass = 1.0 / (jac[p] * w3[i]);

        const double resEx = rhs[0 * npts + p] * invMass;
        const double resEy = rhs[1 * npts + p] * invMass;
        const double resEz = rhs[2 * npts + p] * invMass;
        const double resHx = rhs[3 * npts + p] * invMass;
        const double resHy = rhs[4 * npts + p] * invMass;
        const double resHz = rhs[5 * npts + p] * invMass;

        // ======================================================================
        // OPERATION 2: Low-Storage Runge-Kutta 4(5) Stage Update (Carpenter-Kennedy)
        //   k_comp     = a_s * k_comp + dt * res_scaled_comp
        //   state_comp = state_comp   + b_s * k_comp
        // ======================================================================
        double k0 = a_s * k[0 * npts + p] + dt * resEx;
        double k1 = a_s * k[1 * npts + p] + dt * resEy;
        double k2 = a_s * k[2 * npts + p] + dt * resEz;
        double k3 = a_s * k[3 * npts + p] + dt * resHx;
        double k4 = a_s * k[4 * npts + p] + dt * resHy;
        double k5 = a_s * k[5 * npts + p] + dt * resHz;

        k[0 * npts + p] = k0;
        k[1 * npts + p] = k1;
        k[2 * npts + p] = k2;
        k[3 * npts + p] = k3;
        k[4 * npts + p] = k4;
        k[5 * npts + p] = k5;

        state[0 * stateStride + p] += b_s * k0;
        state[1 * stateStride + p] += b_s * k1;
        state[2 * stateStride + p] += b_s * k2;
        state[3 * stateStride + p] += b_s * k3;
        state[4 * stateStride + p] += b_s * k4;
        state[5 * stateStride + p] += b_s * k5;
    }
}

/**
 * @brief Packs cut face trace values into a contiguous MPI send buffer.
 */
__global__ void gpu_pack_halo_kernel(
    const double* __restrict__ state,
    double* __restrict__ sendBuf,
    const int* __restrict__ sendVolIndices,
    const int* __restrict__ sendBufOffsets,
    const int* __restrict__ exchangeSizes,
    int numHaloPoints, int stateStride)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < numHaloPoints) {
        int v = sendVolIndices[j];
        int base = sendBufOffsets[j];
        int M = exchangeSizes[j];
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            sendBuf[base + c * M] = state[c * stateStride + v];
        }
    }
}

/**
 * @brief Unpacks received MPI buffer into the ghost region of d_state.
 */
__global__ void gpu_unpack_halo_kernel(
    const double* __restrict__ recvBuf,
    double* __restrict__ state,
    const int* __restrict__ recvBufOffsets,
    const int* __restrict__ exchangeSizes,
    int numHaloPoints, int npts, int stateStride)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < numHaloPoints) {
        int base = recvBufOffsets[j];
        int M = exchangeSizes[j];
        int ghostVolIdx = npts + j;
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            state[c * stateStride + ghostVolIdx] = recvBuf[base + c * M];
        }
    }
}

// ==============================================================================
// Launcher Dispatch Wrappers (Exposed C++ API)
// ==============================================================================

namespace nekwave {
namespace device {

void launch_volume_curl(
    const double* Ex, const double* Ey, const double* Ez,
    const double* Hx, const double* Hy, const double* Hz,
    double* resEx, double* resEy, double* resEz,
    double* resHx, double* resHy, double* resHz,
    const double* D, const double* w3_gll, const double* jac,
    const double* rx, const double* sx, const double* tx,
    const double* ry, const double* sy, const double* ty,
    const double* rz, const double* sz, const double* tz,
    int N, int nelt, cudaStream_t stream)
{
    int nxyz = N * N * N;
    int blockSize = nxyz;
    int numBlocks = nelt;
    size_t shmemBytes = (N * N + 3 * nxyz) * sizeof(double);

    gpu_volume_curl_kernel<<<numBlocks, blockSize, shmemBytes, stream>>>(
        Ex, Ey, Ez, Hx, Hy, Hz,
        resEx, resEy, resEz, resHx, resHy, resHz,
        D, w3_gll, jac,
        rx, sx, tx,
        ry, sy, ty,
        rz, sz, tz,
        N, nelt
    );
    NW_GPU_CHECK(cudaGetLastError());
}

void launch_surface_flux(
    const double* d_state,
    const int* d_volIdxMinus, const int* d_volIdxPlus, const int* d_bcType,
    const double* d_nx, const double* d_ny, const double* d_nz, const double* d_dA,
    double* d_rhs, double c0, int N, int totalFacePoints, int npts, int stateStride,
    cudaStream_t stream,
    const double* d_fx, const double* d_fy, const double* d_fz,
    IncidentPlaneWaveConfig incWave, double stageTime)
{
    int blockSize = 256;
    int numBlocks = (totalFacePoints + blockSize - 1) / blockSize;

    gpu_surface_flux_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_state, d_volIdxMinus, d_volIdxPlus, d_bcType,
        d_nx, d_ny, d_nz, d_dA, d_rhs, c0, N, totalFacePoints, npts, stateStride,
        d_fx, d_fy, d_fz, incWave, stageTime
    );
    NW_GPU_CHECK(cudaGetLastError());
}

void launch_pml_step(
    const double* d_state, double* d_rhs,
    double* d_pmlAux, double* d_kPmlAux,
    const int* d_pmlPtr, const double* d_pmlSigma,
    const double* d_jac, const double* d_w3,
    double rk4a, double rk4b, double dt,
    int maxPml, int nxyz, int npts, int stateStride,
    cudaStream_t stream)
{
    if (maxPml <= 0) return;
    int nptsPml = maxPml * nxyz;
    int blockSize = 256;
    int numBlocks = (nptsPml + blockSize - 1) / blockSize;

    gpu_pml_step_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_state, d_rhs, d_pmlAux, d_kPmlAux,
        d_pmlPtr, d_pmlSigma, d_jac, d_w3,
        rk4a, rk4b, dt,
        maxPml, nxyz, npts, stateStride
    );
    NW_GPU_CHECK(cudaGetLastError());
}

void launch_inv_mass_lsrk45_update(
    double* d_state, double* d_k, const double* d_rhs,
    const double* d_jac, const double* d_w3,
    double rk4a, double rk4b, double dt,
    int nelt, int nxyz, int npts, int stateStride,
    cudaStream_t stream)
{
    int blockSize = 256;
    int numBlocks = (npts + blockSize - 1) / blockSize;

    gpu_inv_mass_lsrk45_update_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_state, d_k, d_rhs, d_jac, d_w3,
        rk4a, rk4b, dt,
        nelt, nxyz, npts, stateStride
    );
    NW_GPU_CHECK(cudaGetLastError());
}

void launch_pack_halo(
    const double* d_state, double* d_sendBuf,
    const int* d_sendVolIndices, const int* d_sendBufOffsets,
    const int* d_exchangeSizes, int numHaloPoints, int stateStride,
    cudaStream_t stream)
{
    if (numHaloPoints <= 0) return;
    int blockSize = 256;
    int numBlocks = (numHaloPoints + blockSize - 1) / blockSize;

    gpu_pack_halo_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_state, d_sendBuf, d_sendVolIndices, d_sendBufOffsets,
        d_exchangeSizes, numHaloPoints, stateStride
    );
    NW_GPU_CHECK(cudaGetLastError());
}

void launch_unpack_halo(
    const double* d_recvBuf, double* d_state,
    const int* d_recvBufOffsets, const int* d_exchangeSizes,
    int numHaloPoints, int npts, int stateStride,
    cudaStream_t stream)
{
    if (numHaloPoints <= 0) return;
    int blockSize = 256;
    int numBlocks = (numHaloPoints + blockSize - 1) / blockSize;

    gpu_unpack_halo_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_recvBuf, d_state, d_recvBufOffsets,
        d_exchangeSizes, numHaloPoints, npts, stateStride
    );
    NW_GPU_CHECK(cudaGetLastError());
}

} // namespace device
} // namespace nekwave

