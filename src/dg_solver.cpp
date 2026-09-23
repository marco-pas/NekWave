#include "dg_solver.hpp"
#include "mesh.hpp"
#include "device/dg_kernels.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdio>

// ==============================================================================
// Macro for CUDA Runtime API Error Checking
// ==============================================================================
#ifndef NW_GPU_CHECK
#define NW_GPU_CHECK(call)                                                      \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s (code %d)\n",          \
                         __FILE__, __LINE__, cudaGetErrorString(err), (int)err);\
            std::abort();                                                       \
        }                                                                       \
    } while (0)
#endif

// ==============================================================================
// DgSolver::Impl - Fully GPU-Resident Discontinuous Galerkin Maxwell Engine
// ==============================================================================

class DgSolver::Impl {
public:
    Impl() : bInitialized_(false), stream_(nullptr) {}

    ~Impl() {
        freeBuffers();
    }

    /**
     * @brief Allocates GPU device memory and uploads static mesh metrics and face topology.
     *
     * Prepares device memory residency:
     *   1. State & auxiliary vectors: E, H, k (LSRK45 auxiliary), rhs (spatial residual)
     *   2. 1D GLL differentiation matrix D (N x N) and 3D GLL weights w3 (N^3)
     *   3. Geometric Jacobian J and metric coordinate derivatives (rx, sx, tx, ry, sy, ty, rz, sz, tz)
     *   4. Boundary and interface face quadrature points, unit normals, and area metrics dA
     *   5. Scratch face trace arrays (fEN, fHN) and surface numerical flux arrays (flux)
     */
    bool initialize(const Mesh& mesh, double c0) {
        freeBuffers();

        N_ = mesh.getN();
        nelt_ = mesh.getNumElements();
        nxyz_ = mesh.getNumPointsPerElement();
        npts_ = mesh.getTotalPoints();
        totalEntries_ = 6 * npts_; // 3 electric + 3 magnetic components
        c0_ = c0;                  // 0.0 = Central (conservative), 1.0 = Upwind (dissipative)

        // ----------------------------------------------------------------------
        // Low-Storage Runge-Kutta 4th-Order 5-Stage Scheme (LSRK45)
        // From Carpenter & Kennedy (1994), matching NekCEM's rk_storage in cem_common.F
        // ----------------------------------------------------------------------
        // 5-stage low-storage a-coefficients
        rk4a_ = {0.0,
                 -567301805773.0 / 1357537059087.0,
                 -2404267990393.0 / 2016746695238.0,
                 -3550918686646.0 / 2091501179385.0,
                 -1275806237668.0 / 842570457699.0};

        // 5-stage low-storage b-coefficients
        rk4b_ = {1432997174477.0 / 9575080441755.0,
                 5161836677717.0 / 13612068292357.0,
                 1720146321549.0 / 2090206949498.0,
                 3134564353537.0 / 4481467310338.0,
                 2277821191437.0 / 14882151754819.0};

        // ----------------------------------------------------------------------
        // Flatten 2D Face Quadrature Geometry for Coalesced GPU Streaming
        // ----------------------------------------------------------------------
        // Nanson's relation transforms reference normals to physical normals:
        //      n * dA = J * J^{-T} * n_ref * dA_ref
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

        // ----------------------------------------------------------------------
        // 1. Allocate State, Auxiliary, and Residual GPU Buffers
        // ----------------------------------------------------------------------
        NW_GPU_CHECK(cudaMalloc(&d_state_, totalEntries_ * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_k_, totalEntries_ * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_rhs_, totalEntries_ * sizeof(double)));
        NW_GPU_CHECK(cudaMemsetAsync(d_k_, 0, totalEntries_ * sizeof(double), stream_));

        // ----------------------------------------------------------------------
        // 2. Allocate & Copy 1D GLL Differentiation Matrix D & Quadrature Weights
        // ----------------------------------------------------------------------
        NW_GPU_CHECK(cudaMalloc(&d_D_, mesh.getD().size() * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_D_, mesh.getD().data(), mesh.getD().size() * sizeof(double), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_w3_, mesh.getW3().size() * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_w3_, mesh.getW3().data(), mesh.getW3().size() * sizeof(double), cudaMemcpyHostToDevice, stream_));

        // ----------------------------------------------------------------------
        // 3. Allocate & Copy Volume Coordinate Metrics and Element Jacobians
        // ----------------------------------------------------------------------
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

        // ----------------------------------------------------------------------
        // 4. Allocate & Copy Face Quadrature Topologies and Metric Areas
        // ----------------------------------------------------------------------
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

        // ----------------------------------------------------------------------
        // 5. Allocate Scratch Face Trace and Numerical Flux Buffers
        // ----------------------------------------------------------------------
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

    /**
     * @brief Advances the complete 3D Maxwell system by dt using 5-stage LSRK45.
     *
     * Semi-discrete Maxwell curl equations:
     *      dE/dt = +(1/eps) * curl(H) - (1/eps) * M^{-1} [ F^*_E ]
     *      dH/dt = -(1/mu)  * curl(E) - (1/mu)  * M^{-1} [ F^*_H ]
     *
     * In each stage:
     *   1) Volume curl evaluated directly in shared-memory on chip (+curl(H), -curl(E))
     *   2) Interior fields restricted to element face quadrature nodes (fEN, fHN)
     *   3) Numerical surface Riemann fluxes computed (Central vs Upwind + PEC)
     *   4) Numerical surface fluxes lifted into volume residuals (atomic additions)
     *   5) Scaled by exact diagonal inverse mass matrix: 1 / (J * w_i * w_j * w_k)
     *   6) LSRK45 state & auxiliary vector update:
     *          k = a_s * k + dt * rhs
     *          state = state + b_s * k
     *
     * Zero host-device transfers occur during this method.
     */
    void step(double dt, double time) {
        (void)time;
        assert(bInitialized_);

        // Pointers into the 6 field components of state_
        const double* Ex = d_state_ + 0 * npts_;
        const double* Ey = d_state_ + 1 * npts_;
        const double* Ez = d_state_ + 2 * npts_;
        const double* Hx = d_state_ + 3 * npts_;
        const double* Hy = d_state_ + 4 * npts_;
        const double* Hz = d_state_ + 5 * npts_;

        // Pointers into the 6 field components of rhs_
        double* resEx = d_rhs_ + 0 * npts_;
        double* resEy = d_rhs_ + 1 * npts_;
        double* resEz = d_rhs_ + 2 * npts_;
        double* resHx = d_rhs_ + 3 * npts_;
        double* resHy = d_rhs_ + 4 * npts_;
        double* resHz = d_rhs_ + 5 * npts_;

        // Advance through the 5 stages of Low-Storage Runge-Kutta (LSRK45)
        for (int stage = 0; stage < 5; ++stage) {
            // ------------------------------------------------------------------
            // Step 1: Volume curl of H for Electric field residual: +curl(H)
            // Evaluates dE/dt = +(1/eps) * curl(H) in reference curvilinear coordinates
            // ------------------------------------------------------------------
            nekwave::device::launch_volume_curl(
                Hx, Hy, Hz, resEx, resEy, resEz,
                d_D_, d_w3_, d_jac_,
                d_rx_, d_sx_, d_tx_,
                d_ry_, d_sy_, d_ty_,
                d_rz_, d_sz_, d_tz_,
                1.0, N_, nelt_, stream_
            );

            // ------------------------------------------------------------------
            // Step 2: Volume curl of E for Magnetic field residual: -curl(E)
            // Evaluates dH/dt = -(1/mu) * curl(E) with negative sign
            // ------------------------------------------------------------------
            nekwave::device::launch_volume_curl(
                Ex, Ey, Ez, resHx, resHy, resHz,
                d_D_, d_w3_, d_jac_,
                d_rx_, d_sx_, d_tx_,
                d_ry_, d_sy_, d_ty_,
                d_rz_, d_sz_, d_tz_,
                -1.0, N_, nelt_, stream_
            );

            // ------------------------------------------------------------------
            // Step 3: Restrict volume solution fields to face traces (fEN, fHN)
            // Corresponds to cem_maxwell_restrict_to_face in NekCEM
            // ------------------------------------------------------------------
            nekwave::device::launch_restrict_faces(
                d_state_, d_fEN_, d_fHN_, d_volIdxMinus_,
                totalFacePoints_, npts_, stream_
            );

            // ------------------------------------------------------------------
            // Step 4: Compute numerical surface fluxes (Central or Upwind + PEC)
            // Evaluates jumps [[E]] = E^+ - E^-, [[H]] = H^+ - H^-
            // Upwind/Central numerical flux matching NekCEM lines 991-996:
            //   F^*_H = -0.5 * (n x [[E]]) - 0.5 * C0 * (n x (n x [[H]]))
            //   F^*_E = +0.5 * (n x [[H]]) - 0.5 * C0 * (n x (n x [[E]]))
            // PEC mirror boundary conditions:
            //   n x E^+ = -n x E^- => [[E]] = -2 E^-,  [[H]] = 0
            // ------------------------------------------------------------------
            nekwave::device::launch_compute_flux(
                d_state_, d_fEN_, d_fHN_, d_volIdxPlus_, d_isPEC_,
                d_nx_, d_ny_, d_nz_, d_flux_, c0_, totalFacePoints_, npts_, stream_
            );

            // ------------------------------------------------------------------
            // Step 5: Lift numerical surface fluxes into volume residuals
            // Integrates surface term via face quadrature metric area dA:
            //   res += dA * F^*
            // Corresponds to cem_maxwell_add_flux_to_res in NekCEM
            // ------------------------------------------------------------------
            nekwave::device::launch_add_flux(
                d_flux_, d_volIdxMinus_, d_dA_, d_rhs_,
                totalFacePoints_, npts_, stream_
            );

            // ------------------------------------------------------------------
            // Step 6: Multiply residual by diagonal inverse mass matrix M^{-1}
            // Exact diagonal mass matrix scaling (GLL quadrature mass lumping):
            //   res_E /= (eps * J * w_i * w_j * w_k)
            //   res_H /= (mu  * J * w_i * w_j * w_k)
            // Corresponds to cem_maxwell_invqmass in NekCEM
            // ------------------------------------------------------------------
            nekwave::device::launch_inv_mass(
                d_rhs_, d_jac_, d_w3_, nelt_, nxyz_, npts_, stream_
            );

            // ------------------------------------------------------------------
            // Step 7: Update LSRK45 state & auxiliary vector 'k'
            // Carpenter & Kennedy (1994) low-storage state update:
            //   k = a_s * k + dt * rhs
            //   state = state + b_s * k
            // ------------------------------------------------------------------
            nekwave::device::launch_lsrk45_update(
                d_state_, d_k_, d_rhs_, rk4a_[stage], rk4b_[stage], dt, totalEntries_, stream_
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

    // Device memory buffer pointers
    double* d_state_ = nullptr;
    double* d_k_ = nullptr;
    double* d_rhs_ = nullptr;

    double* d_D_ = nullptr;
    double* d_w3_ = nullptr;
    double* d_jac_ = nullptr;

    double* d_rx_ = nullptr; double* d_sx_ = nullptr; double* d_tx_ = nullptr;
    double* d_ry_ = nullptr; double* d_sy_ = nullptr; double* d_ty_ = nullptr;
    double* d_rz_ = nullptr; double* d_sz_ = nullptr; double* d_tz_ = nullptr;

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
// DgSolver Public Facade Implementation
// ==============================================================================

DgSolver::DgSolver() : impl_(new Impl()) {}

DgSolver::~DgSolver() = default;

bool DgSolver::initialize(const Mesh& mesh, double c0) {
    return impl_->initialize(mesh, c0);
}

void DgSolver::uploadState(const double* hostState, size_t size) {
    impl_->uploadState(hostState, size);
}

void DgSolver::downloadState(double* hostState, size_t size) const {
    impl_->downloadState(hostState, size);
}

void DgSolver::step(double dt, double time) {
    impl_->step(dt, time);
}

void DgSolver::finalize() {
    impl_->freeBuffers();
}
