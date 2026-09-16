#include "gpu_solver.hpp"
#include "mesh.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdio>

#define NW_GPU_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "[NekWave GPU ERROR] %s at %s:%d\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

// ==============================================================================
// CUDA Device Kernels for Maxwell DG-SEM Spatial Discretization & Time Stepping
// ==============================================================================

/**
 * @brief Fused 3D tensor contraction and weighted physical curl kernel.
 *
 * One block per hexahedral element; N^3 threads per block (one per GLL node).
 * Derivative matrix D and element fields u1, u2, u3 are loaded into shared memory.
 * Reference derivatives are computed via register contractions, scaled with metric
 * factors, assembled into physical curl components, and stored directly into residual.
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

    double* s_D  = s_mem;
    double* s_u1 = s_D + N2;
    double* s_u2 = s_u1 + N3;
    double* s_u3 = s_u2 + N3;

    int e = blockIdx.x;
    if (e >= nelt) return;

    int m = threadIdx.x; // Local collocation node index within element [0, N3 - 1]
    int p = e * N3 + m;  // Global collocation node index

    // Cooperatively load 1D derivative matrix D into shared memory
    if (m < N2) {
        s_D[m] = D[m];
    }

    // Load element fields into shared memory
    if (m < N3) {
        s_u1[m] = u1_all[p];
        s_u2[m] = u2_all[p];
        s_u3[m] = u3_all[p];
    }

    __syncthreads();

    if (m >= N3) return;

    // Collocation 3D coordinates (i, j, k)
    int i = m % N;
    int j = (m / N) % N;
    int k = m / N2;

    // Register-accumulated tensor-product reference derivatives
    double u1r = 0.0, u1s = 0.0, u1t = 0.0;
    double u2r = 0.0, u2s = 0.0, u2t = 0.0;
    double u3r = 0.0, u3s = 0.0, u3t = 0.0;

    #pragma unroll
    for (int l = 0; l < N; ++l) {
        double d_il = s_D[i + l * N];
        double d_jl = s_D[j + l * N];
        double d_kl = s_D[k + l * N];

        // Contraction 1: ur = (I (x) I (x) D) u
        int idx_r = l + j * N + k * N2;
        u1r += d_il * s_u1[idx_r];
        u2r += d_il * s_u2[idx_r];
        u3r += d_il * s_u3[idx_r];

        // Contraction 2: us = (I (x) D (x) I) u
        int idx_s = i + l * N + k * N2;
        u1s += d_jl * s_u1[idx_s];
        u2s += d_jl * s_u2[idx_s];
        u3s += d_jl * s_u3[idx_s];

        // Contraction 3: ut = (D (x) I (x) I) u
        int idx_t = i + j * N + l * N2;
        u1t += d_kl * s_u1[idx_t];
        u2t += d_kl * s_u2[idx_t];
        u3t += d_kl * s_u3[idx_t];
    }

    // Quadrature and Jacobian volume weight
    double w = w3_gll[m] * jac[p];

    double u1rw = u1r * w;
    double u1sw = u1s * w;
    double u1tw = u1t * w;

    double u2rw = u2r * w;
    double u2sw = u2s * w;
    double u2tw = u2t * w;

    double u3rw = u3r * w;
    double u3sw = u3s * w;
    double u3tw = u3t * w;

    double rx_k = rx[p], sx_k = sx[p], tx_k = tx[p];
    double ry_k = ry[p], sy_k = sy[p], ty_k = ty[p];
    double rz_k = rz[p], sz_k = sz[p], tz_k = tz[p];

    // Assembly of physical curl components via cofactor metric transformation
    double w1 = (u3rw * ry_k + u3sw * sy_k + u3tw * ty_k)
              - (u2rw * rz_k + u2sw * sz_k + u2tw * tz_k);

    double w2 = (u1rw * rz_k + u1sw * sz_k + u1tw * tz_k)
              - (u3rw * rx_k + u3sw * sx_k + u3tw * tx_k);

    double w3 = (u2rw * rx_k + u2sw * sx_k + u2tw * tx_k)
              - (u1rw * ry_k + u1sw * sy_k + u1tw * ty_k);

    w1_all[p] = sign * w1;
    w2_all[p] = sign * w2;
    w3_all[p] = sign * w3;
}

/**
 * @brief Restricts interior volume fields to element face quadrature nodes.
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
        fEN[0 * totalFacePoints + p] = state[0 * npts + vM];
        fEN[1 * totalFacePoints + p] = state[1 * npts + vM];
        fEN[2 * totalFacePoints + p] = state[2 * npts + vM];

        fHN[0 * totalFacePoints + p] = state[3 * npts + vM];
        fHN[1 * totalFacePoints + p] = state[4 * npts + vM];
        fHN[2 * totalFacePoints + p] = state[5 * npts + vM];
    }
}

/**
 * @brief Computes upwind / central numerical flux on element interface quadrature nodes.
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
        double Ex_m = fEN[0 * totalFacePoints + p];
        double Ey_m = fEN[1 * totalFacePoints + p];
        double Ez_m = fEN[2 * totalFacePoints + p];

        double Hx_m = fHN[0 * totalFacePoints + p];
        double Hy_m = fHN[1 * totalFacePoints + p];
        double Hz_m = fHN[2 * totalFacePoints + p];

        double dEx, dEy, dEz, dHx, dHy, dHz;
        if (isPEC[p] || volIdxPlus[p] < 0) {
            dEx = -2.0 * Ex_m;
            dEy = -2.0 * Ey_m;
            dEz = -2.0 * Ez_m;
            dHx = 0.0;
            dHy = 0.0;
            dHz = 0.0;
        } else {
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

        // Upwind / Central Numerical Flux formula
        double flxHx = -0.5 * nxE_x - 0.5 * c0 * nxnxH_x;
        double flxHy = -0.5 * nxE_y - 0.5 * c0 * nxnxH_y;
        double flxHz = -0.5 * nxE_z - 0.5 * c0 * nxnxH_z;

        double flxEx = +0.5 * nxH_x - 0.5 * c0 * nxnxE_x;
        double flxEy = +0.5 * nxH_y - 0.5 * c0 * nxnxE_y;
        double flxEz = +0.5 * nxH_z - 0.5 * c0 * nxnxE_z;

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
        double a = dA[p];

        atomicAdd(&rhs[0 * npts + vM], a * flux[3 * totalFacePoints + p]); // resEx
        atomicAdd(&rhs[1 * npts + vM], a * flux[4 * totalFacePoints + p]); // resEy
        atomicAdd(&rhs[2 * npts + vM], a * flux[5 * totalFacePoints + p]); // resEz
        atomicAdd(&rhs[3 * npts + vM], a * flux[0 * totalFacePoints + p]); // resHx
        atomicAdd(&rhs[4 * npts + vM], a * flux[1 * totalFacePoints + p]); // resHy
        atomicAdd(&rhs[5 * npts + vM], a * flux[2 * totalFacePoints + p]); // resHz
    }
}

/**
 * @brief Multiplies residuals by diagonal inverse mass matrix (GLL quadrature mass lumping).
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

        const double eps = 1.0;
        const double mu = 1.0;
        double invMass_E = 1.0 / (eps * jac[k] * w3[i]);
        double invMass_H = 1.0 / (mu * jac[k] * w3[i]);

        rhs[0 * npts + k] *= invMass_E;
        rhs[1 * npts + k] *= invMass_E;
        rhs[2 * npts + k] *= invMass_E;
        rhs[3 * npts + k] *= invMass_H;
        rhs[4 * npts + k] *= invMass_H;
        rhs[5 * npts + k] *= invMass_H;
    }
}

/**
 * @brief Vectorized Low-Storage Runge-Kutta 4th-order 5-stage state update.
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
// GpuSolver::Impl Implementation
// ==============================================================================

class GpuSolver::Impl {
public:
    Impl() : bInitialized_(false), stream_(nullptr) {}

    ~Impl() {
        freeBuffers();
    }

    bool initialize(const Mesh& mesh, double c0) {
        freeBuffers();

        N_ = mesh.getN();
        nelt_ = mesh.getNumElements();
        nxyz_ = mesh.getNumPointsPerElement();
        npts_ = mesh.getTotalPoints();
        totalEntries_ = 6 * npts_;
        c0_ = c0;

        // Initialize Low-Storage Runge-Kutta coefficients
        rk4a_ = {0.0,
                 -567301805773.0 / 1357537059087.0,
                 -2404267990393.0 / 2016746695238.0,
                 -3550918686646.0 / 2091501179385.0,
                 -1275806237668.0 / 842570457699.0};

        rk4b_ = {1432997174477.0 / 9575080441755.0,
                 5161836677717.0 / 13612068292357.0,
                 1720146321549.0 / 2090206949498.0,
                 3134564353537.0 / 4481467310338.0,
                 2277821191437.0 / 14882151754819.0};

        // Flatten face data
        const auto& faceData = mesh.getFaceData();
        totalFacePoints_ = 0;
        for (const auto& fd : faceData) {
            totalFacePoints_ += fd.points.size();
        }

        std::vector<int> h_volIdxMinus(totalFacePoints_);
        std::vector<int> h_volIdxPlus(totalFacePoints_);
        std::vector<int> h_isPEC(totalFacePoints_);
        std::vector<double> h_nx(totalFacePoints_);
        std::vector<double> h_ny(totalFacePoints_);
        std::vector<double> h_nz(totalFacePoints_);
        std::vector<double> h_dA(totalFacePoints_);

        size_t ptIdx = 0;
        for (const auto& fd : faceData) {
            int pecFlag = (fd.bcType == "PEC") ? 1 : 0;
            for (const auto& pt : fd.points) {
                h_volIdxMinus[ptIdx] = pt.volIdxMinus;
                h_volIdxPlus[ptIdx]  = pt.volIdxPlus;
                h_isPEC[ptIdx]       = pecFlag || (pt.volIdxPlus < 0);
                h_nx[ptIdx]          = pt.nx;
                h_ny[ptIdx]          = pt.ny;
                h_nz[ptIdx]          = pt.nz;
                h_dA[ptIdx]          = pt.dA;
                ptIdx++;
            }
        }

        NW_GPU_CHECK(cudaStreamCreate(&stream_));

        // 1. Allocate state & RK vector buffers
        NW_GPU_CHECK(cudaMalloc(&d_state_, totalEntries_ * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_k_, totalEntries_ * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_rhs_, totalEntries_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemsetAsync(d_k_, 0, totalEntries_ * sizeof(double), stream_));

        // 2. Allocate & copy 1D differentiation matrix & quadrature weights
        NW_GPU_CHECK(cudaMalloc(&d_D_, mesh.getD().size() * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_D_, mesh.getD().data(), mesh.getD().size() * sizeof(double), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_w3_, mesh.getW3().size() * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_w3_, mesh.getW3().data(), mesh.getW3().size() * sizeof(double), cudaMemcpyHostToDevice, stream_));

        // 3. Allocate & copy volume metrics
        NW_GPU_CHECK(cudaMalloc(&d_jac_, npts_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_jac_, mesh.getJac().data(), npts_ * sizeof(double), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_rx_, npts_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_rx_, mesh.getRx().data(), npts_ * sizeof(double), cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMalloc(&d_sx_, npts_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_sx_, mesh.getSx().data(), npts_ * sizeof(double), cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMalloc(&d_tx_, npts_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_tx_, mesh.getTx().data(), npts_ * sizeof(double), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_ry_, npts_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_ry_, mesh.getRy().data(), npts_ * sizeof(double), cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMalloc(&d_sy_, npts_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_sy_, mesh.getSy().data(), npts_ * sizeof(double), cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMalloc(&d_ty_, npts_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_ty_, mesh.getTy().data(), npts_ * sizeof(double), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_rz_, npts_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_rz_, mesh.getRz().data(), npts_ * sizeof(double), cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMalloc(&d_sz_, npts_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_sz_, mesh.getSz().data(), npts_ * sizeof(double), cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMalloc(&d_tz_, npts_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_tz_, mesh.getTz().data(), npts_ * sizeof(double), cudaMemcpyHostToDevice, stream_));

        // 4. Allocate & copy face data
        NW_GPU_CHECK(cudaMalloc(&d_volIdxMinus_, totalFacePoints_ * sizeof(int)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_volIdxMinus_, h_volIdxMinus.data(), totalFacePoints_ * sizeof(int), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_volIdxPlus_, totalFacePoints_ * sizeof(int)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_volIdxPlus_, h_volIdxPlus.data(), totalFacePoints_ * sizeof(int), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_isPEC_, totalFacePoints_ * sizeof(int)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_isPEC_, h_isPEC.data(), totalFacePoints_ * sizeof(int), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_nx_, totalFacePoints_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_nx_, h_nx.data(), totalFacePoints_ * sizeof(double), cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMalloc(&d_ny_, totalFacePoints_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_ny_, h_ny.data(), totalFacePoints_ * sizeof(double), cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMalloc(&d_nz_, totalFacePoints_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_nz_, h_nz.data(), totalFacePoints_ * sizeof(double), cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMalloc(&d_dA_, totalFacePoints_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_dA_, h_dA.data(), totalFacePoints_ * sizeof(double), cudaMemcpyHostToDevice, stream_));

        // 5. Allocate scratch face buffers
        NW_GPU_CHECK(cudaMalloc(&d_fEN_, 3 * totalFacePoints_ * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_fHN_, 3 * totalFacePoints_ * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_flux_, 6 * totalFacePoints_ * sizeof(double)));

        NW_GPU_CHECK(cudaStreamSynchronize(stream_));
        bInitialized_ = true;
        return true;
    }

    void uploadState(const double* hostState, size_t size) {
        assert(bInitialized_ && size == static_cast<size_t>(totalEntries_));
        NW_GPU_CHECK(cudaMemcpyAsync(d_state_, hostState, size * sizeof(double), cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaStreamSynchronize(stream_));
    }

    void downloadState(double* hostState, size_t size) const {
        assert(bInitialized_ && size == static_cast<size_t>(totalEntries_));
        NW_GPU_CHECK(cudaMemcpyAsync(hostState, d_state_, size * sizeof(double), cudaMemcpyDeviceToHost, stream_));
        NW_GPU_CHECK(cudaStreamSynchronize(stream_));
    }

    void step(double dt, double time) {
        (void)time;
        assert(bInitialized_);

        // Configure kernel launch dimensions
        int blockSizeCurl = nxyz_;
        int numBlocksCurl = nelt_;
        size_t shmemBytes = (N_ * N_ + 3 * nxyz_) * sizeof(double);

        int blockSize1D = 256;
        int numBlocksFaces = (totalFacePoints_ + blockSize1D - 1) / blockSize1D;
        int numBlocksMass = (nelt_ * nxyz_ + blockSize1D - 1) / blockSize1D;
        int numBlocksUpdate = (totalEntries_ + blockSize1D - 1) / blockSize1D;

        const double* Ex = d_state_ + 0 * npts_;
        const double* Ey = d_state_ + 1 * npts_;
        const double* Ez = d_state_ + 2 * npts_;
        const double* Hx = d_state_ + 3 * npts_;
        const double* Hy = d_state_ + 4 * npts_;
        const double* Hz = d_state_ + 5 * npts_;

        double* resEx = d_rhs_ + 0 * npts_;
        double* resEy = d_rhs_ + 1 * npts_;
        double* resEz = d_rhs_ + 2 * npts_;
        double* resHx = d_rhs_ + 3 * npts_;
        double* resHy = d_rhs_ + 4 * npts_;
        double* resHz = d_rhs_ + 5 * npts_;

        // Advance through 5 stages of LSRK45
        for (int stage = 0; stage < 5; ++stage) {
            // Step 1: Volume curl of H for E residual: curl(H) with sign = +1.0
            gpu_volume_curl_kernel<<<numBlocksCurl, blockSizeCurl, shmemBytes, stream_>>>(
                Hx, Hy, Hz, resEx, resEy, resEz,
                d_D_, d_w3_, d_jac_,
                d_rx_, d_sx_, d_tx_,
                d_ry_, d_sy_, d_ty_,
                d_rz_, d_sz_, d_tz_,
                1.0, N_, nelt_
            );

            // Step 2: Volume curl of E for H residual: -curl(E) with sign = -1.0
            gpu_volume_curl_kernel<<<numBlocksCurl, blockSizeCurl, shmemBytes, stream_>>>(
                Ex, Ey, Ez, resHx, resHy, resHz,
                d_D_, d_w3_, d_jac_,
                d_rx_, d_sx_, d_tx_,
                d_ry_, d_sy_, d_ty_,
                d_rz_, d_sz_, d_tz_,
                -1.0, N_, nelt_
            );

            // Step 3: Restrict volume fields to face traces
            gpu_restrict_faces_kernel<<<numBlocksFaces, blockSize1D, 0, stream_>>>(
                d_state_, d_fEN_, d_fHN_, d_volIdxMinus_, totalFacePoints_, npts_
            );

            // Step 4: Compute numerical surface flux
            gpu_compute_flux_kernel<<<numBlocksFaces, blockSize1D, 0, stream_>>>(
                d_state_, d_fEN_, d_fHN_, d_volIdxPlus_, d_isPEC_,
                d_nx_, d_ny_, d_nz_, d_flux_, c0_, totalFacePoints_, npts_
            );

            // Step 5: Lift numerical flux into volume residuals
            gpu_add_flux_kernel<<<numBlocksFaces, blockSize1D, 0, stream_>>>(
                d_flux_, d_volIdxMinus_, d_dA_, d_rhs_, totalFacePoints_, npts_
            );

            // Step 6: Multiply residual by diagonal inverse mass matrix
            gpu_inv_mass_kernel<<<numBlocksMass, blockSize1D, 0, stream_>>>(
                d_rhs_, d_jac_, d_w3_, nelt_, nxyz_, npts_
            );

            // Step 7: Update LSRK45 state & auxiliary vector 'k'
            gpu_lsrk45_update_kernel<<<numBlocksUpdate, blockSize1D, 0, stream_>>>(
                d_state_, d_k_, d_rhs_, rk4a_[stage], rk4b_[stage], dt, totalEntries_
            );
        }

        // Stream synchronization only occurs at the end of the full time step
        NW_GPU_CHECK(cudaStreamSynchronize(stream_));
    }

    void freeBuffers() {
        if (!bInitialized_) return;

        cudaFree(d_state_); d_state_ = nullptr;
        cudaFree(d_k_); d_k_ = nullptr;
        cudaFree(d_rhs_); d_rhs_ = nullptr;

        cudaFree(d_D_); d_D_ = nullptr;
        cudaFree(d_w3_); d_w3_ = nullptr;
        cudaFree(d_jac_); d_jac_ = nullptr;

        cudaFree(d_rx_); d_rx_ = nullptr;
        cudaFree(d_sx_); d_sx_ = nullptr;
        cudaFree(d_tx_); d_tx_ = nullptr;

        cudaFree(d_ry_); d_ry_ = nullptr;
        cudaFree(d_sy_); d_sy_ = nullptr;
        cudaFree(d_ty_); d_ty_ = nullptr;

        cudaFree(d_rz_); d_rz_ = nullptr;
        cudaFree(d_sz_); d_sz_ = nullptr;
        cudaFree(d_tz_); d_tz_ = nullptr;

        cudaFree(d_volIdxMinus_); d_volIdxMinus_ = nullptr;
        cudaFree(d_volIdxPlus_); d_volIdxPlus_ = nullptr;
        cudaFree(d_isPEC_); d_isPEC_ = nullptr;

        cudaFree(d_nx_); d_nx_ = nullptr;
        cudaFree(d_ny_); d_ny_ = nullptr;
        cudaFree(d_nz_); d_nz_ = nullptr;
        cudaFree(d_dA_); d_dA_ = nullptr;

        cudaFree(d_fEN_); d_fEN_ = nullptr;
        cudaFree(d_fHN_); d_fHN_ = nullptr;
        cudaFree(d_flux_); d_flux_ = nullptr;

        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }

        bInitialized_ = false;
    }

private:
    bool bInitialized_;
    cudaStream_t stream_;

    int N_;
    int nelt_;
    int nxyz_;
    int npts_;
    int totalEntries_;
    int totalFacePoints_;
    double c0_;

    std::vector<double> rk4a_;
    std::vector<double> rk4b_;

    // Device pointers: State & RK
    double* d_state_ = nullptr;
    double* d_k_ = nullptr;
    double* d_rhs_ = nullptr;

    // Device pointers: Mesh & metric operators
    double* d_D_ = nullptr;
    double* d_w3_ = nullptr;
    double* d_jac_ = nullptr;

    double* d_rx_ = nullptr;
    double* d_sx_ = nullptr;
    double* d_tx_ = nullptr;

    double* d_ry_ = nullptr;
    double* d_sy_ = nullptr;
    double* d_ty_ = nullptr;

    double* d_rz_ = nullptr;
    double* d_sz_ = nullptr;
    double* d_tz_ = nullptr;

    // Device pointers: Face geometry & traces
    int* d_volIdxMinus_ = nullptr;
    int* d_volIdxPlus_ = nullptr;
    int* d_isPEC_ = nullptr;

    double* d_nx_ = nullptr;
    double* d_ny_ = nullptr;
    double* d_nz_ = nullptr;
    double* d_dA_ = nullptr;

    double* d_fEN_ = nullptr;
    double* d_fHN_ = nullptr;
    double* d_flux_ = nullptr;
};

// ==============================================================================
// GpuSolver Public Interface Trampolines
// ==============================================================================

GpuSolver::GpuSolver() : impl_(new Impl()) {}

GpuSolver::~GpuSolver() = default;

bool GpuSolver::initialize(const Mesh& mesh, double c0) {
    return impl_->initialize(mesh, c0);
}

void GpuSolver::uploadState(const double* hostState, size_t size) {
    impl_->uploadState(hostState, size);
}

void GpuSolver::downloadState(double* hostState, size_t size) const {
    impl_->downloadState(hostState, size);
}

void GpuSolver::step(double dt, double time) {
    impl_->step(dt, time);
}

void GpuSolver::finalize() {
    impl_->freeBuffers();
}
