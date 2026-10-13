#include "dg_solver.hpp"
#include "mesh.hpp"
#include "boundary_conditions.hpp"
#include "comm.hpp"
#include "device/dg_sim_kernels.hpp"
#include "device/dg_preproc_kernels.hpp"
#include "device/gpu_runtime.hpp"

#ifdef NEKWAVE_ENABLE_MPI
#include <mpi.h>
#endif
#ifndef USE_HIP
#include <nvtx3/nvToolsExt.h>
#endif
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdio>
#include <chrono>

// ==============================================================================
// DgSolver::Impl - Fully GPU-Resident Discontinuous Galerkin Maxwell Engine
// ==============================================================================

class DgSolver::Impl {
public:
    Impl() : bInitialized_(false), stream_(nullptr) {}

    ~Impl() {
        freeBuffers();
    }

    bool initialize(const Mesh& mesh, double c0) {
        double dxmin = 0.0;
        return preprocessAndInitialize(const_cast<Mesh&>(mesh), c0, &dxmin, true);
    }

    /**
     * @brief Performs GPU-native mesh preprocessing (volume metrics, face normals/matching,
     *        UPML sigma grading, and CFL min node distance) and allocates all persistent
     *        GPU solver buffers.
     */
    bool preprocessAndInitialize(Mesh& mesh, double c0,
                                 double* outDxmin,
                                 bool downloadHostCoords) {
        using Clock = std::chrono::high_resolution_clock;
        freeBuffers();
        gpuPreprocessTimings_ = GpuPreprocessTimings{};

        N_ = mesh.getN();
        nelt_ = mesh.getNumElements();
        nxyz_ = mesh.getNumPointsPerElement();
        npts_ = mesh.getTotalPoints();
        numHaloPoints_ = mesh.isPartitioned() ? mesh.getNumHaloPoints() : 0;
        stateStride_ = npts_ + numHaloPoints_;
        haloExchanges_ = mesh.getMpiHalos();
        totalEntries_ = 6 * npts_; // 3 electric + 3 magnetic components
        c0_ = c0;                  // 0.0 = Central (conservative), 1.0 = Upwind (dissipative)

        // ----------------------------------------------------------------------
        // Low-Storage Runge-Kutta 4th-Order 5-Stage Scheme (LSRK45)
        // From Carpenter & Kennedy (1994), matching NekCEM's rk_storage in cem_common.F
        // ----------------------------------------------------------------------
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

        rk4c_ = {0.0,
                 1432997174477.0 / 9575080441755.0,
                 2526269341429.0 / 6820363962896.0,
                 2006345519317.0 / 3224310063776.0,
                 2802321613138.0 / 2924317926251.0};

        gpuPreprocessTimings_.usedGpuPreprocess = true;
        const int numFaces = static_cast<int>(mesh.getFaces().size());
        totalFacePoints_ = numFaces * N_ * N_;

        PmlData localPml = mesh.getPmlData();
        if (!localPml.enabled && BoundaryConditions::hasPmlFaces(mesh)) {
            localPml = BoundaryConditions::initializePml(mesh, mesh.getPmlConfig());
        }
        maxPml_ = localPml.maxPml;
        nptsPml_ = maxPml_ * nxyz_;

        NW_GPU_CHECK(cudaStreamCreate(&stream_));

        // ------------------------------------------------------------------
        // Stage A: Allocate persistent GPU buffers & upload compact topology
        //          (Element corners: 24 doubles/elem; Faces: 5 ints/face)
        // ------------------------------------------------------------------
        auto tStage = Clock::now();
        NW_PROFILER_PUSH("GPU::allocAndUploadTopology");

        NW_GPU_CHECK(cudaMalloc(&d_state_, 6 * static_cast<size_t>(stateStride_) * sizeof(double)));
        NW_GPU_CHECK(cudaMemsetAsync(d_state_, 0, 6 * static_cast<size_t>(stateStride_) * sizeof(double), stream_));
        NW_GPU_CHECK(cudaMalloc(&d_k_, static_cast<size_t>(totalEntries_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_rhs_, static_cast<size_t>(totalEntries_) * sizeof(double)));
        NW_GPU_CHECK(cudaMemsetAsync(d_k_, 0, static_cast<size_t>(totalEntries_) * sizeof(double), stream_));

        // Allocate MPI Halo Exchange Buffers & Metadata (if partitioned across ranks)
        if (numHaloPoints_ > 0) {
            exchangeBufOffsets_.resize(haloExchanges_.size());
            int curOffset = 0;
            for (size_t h = 0; h < haloExchanges_.size(); ++h) {
                exchangeBufOffsets_[h] = curOffset;
                curOffset += 6 * haloExchanges_[h].numPoints;
            }

            std::vector<int> h_sendVolIndices(numHaloPoints_);
            std::vector<int> h_sendBufOffsets(numHaloPoints_);
            std::vector<int> h_recvBufOffsets(numHaloPoints_);
            std::vector<int> h_exchangeSizes(numHaloPoints_);

            for (size_t h = 0; h < haloExchanges_.size(); ++h) {
                int base_buf = exchangeBufOffsets_[h];
                int M = haloExchanges_[h].numPoints;
                for (int p = 0; p < M; ++p) {
                    int j = haloExchanges_[h].ghostOffset + p;
                    h_sendVolIndices[j] = haloExchanges_[h].sendVolIndices[p];
                    h_sendBufOffsets[j] = base_buf + p;
                    h_recvBufOffsets[j] = base_buf + p;
                    h_exchangeSizes[j] = M;
                }
            }

            NW_GPU_CHECK(cudaMalloc(&d_sendVolIndices_, static_cast<size_t>(numHaloPoints_) * sizeof(int)));
            NW_GPU_CHECK(cudaMemcpyAsync(d_sendVolIndices_, h_sendVolIndices.data(), static_cast<size_t>(numHaloPoints_) * sizeof(int), cudaMemcpyHostToDevice, stream_));

            NW_GPU_CHECK(cudaMalloc(&d_sendBufOffsets_, static_cast<size_t>(numHaloPoints_) * sizeof(int)));
            NW_GPU_CHECK(cudaMemcpyAsync(d_sendBufOffsets_, h_sendBufOffsets.data(), static_cast<size_t>(numHaloPoints_) * sizeof(int), cudaMemcpyHostToDevice, stream_));

            NW_GPU_CHECK(cudaMalloc(&d_recvBufOffsets_, static_cast<size_t>(numHaloPoints_) * sizeof(int)));
            NW_GPU_CHECK(cudaMemcpyAsync(d_recvBufOffsets_, h_recvBufOffsets.data(), static_cast<size_t>(numHaloPoints_) * sizeof(int), cudaMemcpyHostToDevice, stream_));

            NW_GPU_CHECK(cudaMalloc(&d_exchangeSizes_, static_cast<size_t>(numHaloPoints_) * sizeof(int)));
            NW_GPU_CHECK(cudaMemcpyAsync(d_exchangeSizes_, h_exchangeSizes.data(), static_cast<size_t>(numHaloPoints_) * sizeof(int), cudaMemcpyHostToDevice, stream_));

            NW_GPU_CHECK(cudaMalloc(&d_mpiSendBuf_, 6 * static_cast<size_t>(numHaloPoints_) * sizeof(double)));
            NW_GPU_CHECK(cudaMalloc(&d_mpiRecvBuf_, 6 * static_cast<size_t>(numHaloPoints_) * sizeof(double)));
        }

        NW_GPU_CHECK(cudaMalloc(&d_D_, mesh.getD().size() * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_D_, mesh.getD().data(), mesh.getD().size() * sizeof(double), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_w3_, mesh.getW3().size() * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_w3_, mesh.getW3().data(), mesh.getW3().size() * sizeof(double), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_jac_, static_cast<size_t>(npts_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_rx_, static_cast<size_t>(npts_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_sx_, static_cast<size_t>(npts_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_tx_, static_cast<size_t>(npts_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_ry_, static_cast<size_t>(npts_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_sy_, static_cast<size_t>(npts_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_ty_, static_cast<size_t>(npts_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_rz_, static_cast<size_t>(npts_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_sz_, static_cast<size_t>(npts_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_tz_, static_cast<size_t>(npts_) * sizeof(double)));

        NW_GPU_CHECK(cudaMalloc(&d_volIdxMinus_, static_cast<size_t>(totalFacePoints_) * sizeof(int)));
        NW_GPU_CHECK(cudaMalloc(&d_volIdxPlus_, static_cast<size_t>(totalFacePoints_) * sizeof(int)));
        NW_GPU_CHECK(cudaMalloc(&d_bcType_, static_cast<size_t>(totalFacePoints_) * sizeof(int)));
        NW_GPU_CHECK(cudaMalloc(&d_nx_, static_cast<size_t>(totalFacePoints_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_ny_, static_cast<size_t>(totalFacePoints_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_nz_, static_cast<size_t>(totalFacePoints_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_dA_, static_cast<size_t>(totalFacePoints_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_fx_, static_cast<size_t>(totalFacePoints_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_fy_, static_cast<size_t>(totalFacePoints_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_fz_, static_cast<size_t>(totalFacePoints_) * sizeof(double)));

        // Temporary preprocessing buffers (freed before scratch allocation so peak VRAM is unchanged)
        double* d_corners = nullptr;
        double* d_gll_z = nullptr;
        double* d_gll_w = nullptr;
        double* d_coordX = nullptr;
        double* d_coordY = nullptr;
        double* d_coordZ = nullptr;
        int* d_faceElem = nullptr;
        int* d_faceId = nullptr;
        int* d_neighborElem = nullptr;
        int* d_neighborFace = nullptr;
        int* d_faceBcCode = nullptr;
        int* d_mpiCutGhostIdx = nullptr;
        int* d_pmlTag = nullptr;

        const size_t cornersBytes = static_cast<size_t>(nelt_) * 24 * sizeof(double);
        NW_GPU_CHECK(cudaMalloc(&d_corners, cornersBytes));
        NW_GPU_CHECK(cudaMemcpyAsync(d_corners, mesh.getElementCorners().data(), cornersBytes, cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_gll_z, static_cast<size_t>(N_) * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_gll_z, mesh.getGllZ().data(), static_cast<size_t>(N_) * sizeof(double), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_gll_w, static_cast<size_t>(N_) * sizeof(double)));
        NW_GPU_CHECK(cudaMemcpyAsync(d_gll_w, mesh.getGllW().data(), static_cast<size_t>(N_) * sizeof(double), cudaMemcpyHostToDevice, stream_));

        NW_GPU_CHECK(cudaMalloc(&d_coordX, static_cast<size_t>(npts_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_coordY, static_cast<size_t>(npts_) * sizeof(double)));
        NW_GPU_CHECK(cudaMalloc(&d_coordZ, static_cast<size_t>(npts_) * sizeof(double)));

        const auto& faces = mesh.getFaces();
        std::vector<int> h_faceElem(numFaces);
        std::vector<int> h_faceId(numFaces);
        std::vector<int> h_neighborElem(numFaces);
        std::vector<int> h_neighborFace(numFaces);
        std::vector<int> h_faceBcCode(numFaces);
        for (int f = 0; f < numFaces; ++f) {
            h_faceElem[f] = faces[f].elementId;
            h_faceId[f] = faces[f].faceId;
            h_neighborElem[f] = faces[f].neighborElementId;
            h_neighborFace[f] = faces[f].neighborFaceId;
            h_faceBcCode[f] = static_cast<int>(BoundaryConditions::parseBcTag(faces[f].bcType, faces[f].neighborElementId));
        }

        const size_t faceIntsBytes = static_cast<size_t>(numFaces) * sizeof(int);
        NW_GPU_CHECK(cudaMalloc(&d_faceElem, faceIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_faceId, faceIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_neighborElem, faceIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_neighborFace, faceIntsBytes));
        NW_GPU_CHECK(cudaMalloc(&d_faceBcCode, faceIntsBytes));

        NW_GPU_CHECK(cudaMemcpyAsync(d_faceElem, h_faceElem.data(), faceIntsBytes, cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMemcpyAsync(d_faceId, h_faceId.data(), faceIntsBytes, cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMemcpyAsync(d_neighborElem, h_neighborElem.data(), faceIntsBytes, cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMemcpyAsync(d_neighborFace, h_neighborFace.data(), faceIntsBytes, cudaMemcpyHostToDevice, stream_));
        NW_GPU_CHECK(cudaMemcpyAsync(d_faceBcCode, h_faceBcCode.data(), faceIntsBytes, cudaMemcpyHostToDevice, stream_));

        if (!mesh.getMpiCutGhostIndices().empty()) {
            const size_t cutBytes = mesh.getMpiCutGhostIndices().size() * sizeof(int);
            NW_GPU_CHECK(cudaMalloc(&d_mpiCutGhostIdx, cutBytes));
            NW_GPU_CHECK(cudaMemcpyAsync(d_mpiCutGhostIdx, mesh.getMpiCutGhostIndices().data(), cutBytes, cudaMemcpyHostToDevice, stream_));
        }

        size_t pmlUploadBytes = 0;
        if (maxPml_ > 0) {
            std::vector<int> h_pmlTagCompact(maxPml_);
            for (int e = 0; e < maxPml_; ++e) {
                int ie = localPml.pmlPtr[e];
                h_pmlTagCompact[e] = (ie >= 0 && ie < static_cast<int>(localPml.pmlTag.size())) ? localPml.pmlTag[ie] : 0;
            }
            NW_GPU_CHECK(cudaMalloc(&d_pmlPtr_, static_cast<size_t>(maxPml_) * sizeof(int)));
            NW_GPU_CHECK(cudaMemcpyAsync(d_pmlPtr_, localPml.pmlPtr.data(), static_cast<size_t>(maxPml_) * sizeof(int), cudaMemcpyHostToDevice, stream_));

            NW_GPU_CHECK(cudaMalloc(&d_pmlTag, static_cast<size_t>(maxPml_) * sizeof(int)));
            NW_GPU_CHECK(cudaMemcpyAsync(d_pmlTag, h_pmlTagCompact.data(), static_cast<size_t>(maxPml_) * sizeof(int), cudaMemcpyHostToDevice, stream_));
            pmlUploadBytes = 2 * static_cast<size_t>(maxPml_) * sizeof(int);

            NW_GPU_CHECK(cudaMalloc(&d_pmlSigma_, 3 * static_cast<size_t>(npts_) * sizeof(double)));
            NW_GPU_CHECK(cudaMemsetAsync(d_pmlSigma_, 0, 3 * static_cast<size_t>(npts_) * sizeof(double), stream_));
        }

        NW_GPU_CHECK(cudaStreamSynchronize(stream_));
        NW_PROFILER_POP();
        gpuPreprocessTimings_.topoUploadSec = std::chrono::duration<double>(Clock::now() - tStage).count();
        gpuPreprocessTimings_.h2dTopologyBytes = cornersBytes + 5 * faceIntsBytes + pmlUploadBytes +
                                                 (mesh.getD().size() + mesh.getW3().size() + 2 * static_cast<size_t>(N_)) * sizeof(double);

        // ------------------------------------------------------------------
        // Stage B: GPU Kernel - Compute 3D GLL Coordinates, Jacobian, Metrics
        // ------------------------------------------------------------------
        tStage = Clock::now();
        NW_PROFILER_PUSH("GPU::computeVolumeMetrics");
        nekwave::device::launch_compute_volume_metrics(
            d_corners, d_gll_z,
            d_coordX, d_coordY, d_coordZ,
            d_jac_,
            d_rx_, d_sx_, d_tx_,
            d_ry_, d_sy_, d_ty_,
            d_rz_, d_sz_, d_tz_,
            N_, nelt_, npts_, stream_
        );
        NW_GPU_CHECK(cudaStreamSynchronize(stream_));
        NW_PROFILER_POP();
        gpuPreprocessTimings_.volumeMetricsKernelSec = std::chrono::duration<double>(Clock::now() - tStage).count();

        cudaFree(d_corners);
        d_corners = nullptr;

        // ------------------------------------------------------------------
        // Stage C: GPU Kernel - Face Normals, Area dA, and D4 Neighbor Match
        // ------------------------------------------------------------------
        tStage = Clock::now();
        NW_PROFILER_PUSH("GPU::setupFaceData");
        nekwave::device::launch_setup_face_data(
            d_faceElem, d_faceId, d_neighborElem, d_neighborFace, d_faceBcCode,
            d_gll_w,
            d_coordX, d_coordY, d_coordZ,
            d_jac_,
            d_rx_, d_sx_, d_tx_,
            d_ry_, d_sy_, d_ty_,
            d_rz_, d_sz_, d_tz_,
            d_volIdxMinus_, d_volIdxPlus_, d_bcType_,
            d_nx_, d_ny_, d_nz_, d_dA_,
            d_fx_, d_fy_, d_fz_,
            N_, numFaces, totalFacePoints_, stream_,
            d_mpiCutGhostIdx
        );
        NW_GPU_CHECK(cudaStreamSynchronize(stream_));
        NW_PROFILER_POP();
        gpuPreprocessTimings_.faceSetupKernelSec = std::chrono::duration<double>(Clock::now() - tStage).count();

        if (d_mpiCutGhostIdx) { cudaFree(d_mpiCutGhostIdx); d_mpiCutGhostIdx = nullptr; }
        cudaFree(d_faceElem); d_faceElem = nullptr;
        cudaFree(d_faceId); d_faceId = nullptr;
        cudaFree(d_neighborElem); d_neighborElem = nullptr;
        cudaFree(d_neighborFace); d_neighborFace = nullptr;
        cudaFree(d_faceBcCode); d_faceBcCode = nullptr;
        cudaFree(d_gll_w); d_gll_w = nullptr;

        // ------------------------------------------------------------------
        // Stage D: GPU Kernel - UPML Damping Conductivity Profiles (pmlSigma)
        // ------------------------------------------------------------------
        if (maxPml_ > 0) {
            tStage = Clock::now();
            NW_PROFILER_PUSH("GPU::setupPmlSigma");
            nekwave::device::launch_compute_pml_sigma(
                d_pmlPtr_, d_pmlTag,
                d_coordX, d_coordY, d_coordZ,
                localPml.pmlInner.data(), localPml.pmlOuter.data(),
                static_cast<double>(mesh.getPmlConfig().order),
                mesh.getPmlConfig().reflectErr,
                d_pmlSigma_,
                maxPml_, nxyz_, npts_, stream_
            );
            NW_GPU_CHECK(cudaStreamSynchronize(stream_));
            NW_PROFILER_POP();
            gpuPreprocessTimings_.pmlSigmaKernelSec = std::chrono::duration<double>(Clock::now() - tStage).count();

            cudaFree(d_pmlTag);
            d_pmlTag = nullptr;
        }

        // ------------------------------------------------------------------
        // Stage E: GPU Kernel - Minimum GLL Node Spacing (dxmin) for CFL
        // ------------------------------------------------------------------
        tStage = Clock::now();
        NW_PROFILER_PUSH("GPU::computeMinNodeDistance");
        const double dxiMin = (N_ > 1) ? (mesh.getGllZ()[1] - mesh.getGllZ()[0]) : 2.0;
        const double dxminVal = nekwave::device::launch_compute_min_node_distance(
            d_coordX, d_coordY, d_coordZ,
            d_rx_, d_sx_, d_tx_,
            d_ry_, d_sy_, d_ty_,
            d_rz_, d_sz_, d_tz_,
            dxiMin, N_, nelt_, npts_, stream_
        );
        mesh.setMinNodeDistance(dxminVal);
        if (outDxmin != nullptr) {
            *outDxmin = dxminVal;
        }
        NW_PROFILER_POP();
        gpuPreprocessTimings_.minDistKernelSec = std::chrono::duration<double>(Clock::now() - tStage).count();

        // ------------------------------------------------------------------
        // Stage F: Optional Host Coordinate/Jacobian Download (only if needed)
        // ------------------------------------------------------------------
        if (downloadHostCoords && !mesh.hasHostMetrics()) {
            tStage = Clock::now();
            NW_PROFILER_PUSH("GPU::downloadHostCoords");
            std::vector<double> cx(npts_), cy(npts_), cz(npts_), jac(npts_);
            NW_GPU_CHECK(cudaMemcpyAsync(cx.data(), d_coordX, static_cast<size_t>(npts_) * sizeof(double), cudaMemcpyDeviceToHost, stream_));
            NW_GPU_CHECK(cudaMemcpyAsync(cy.data(), d_coordY, static_cast<size_t>(npts_) * sizeof(double), cudaMemcpyDeviceToHost, stream_));
            NW_GPU_CHECK(cudaMemcpyAsync(cz.data(), d_coordZ, static_cast<size_t>(npts_) * sizeof(double), cudaMemcpyDeviceToHost, stream_));
            NW_GPU_CHECK(cudaMemcpyAsync(jac.data(), d_jac_, static_cast<size_t>(npts_) * sizeof(double), cudaMemcpyDeviceToHost, stream_));
            NW_GPU_CHECK(cudaStreamSynchronize(stream_));
            mesh.setHostCoordsAndJac(std::move(cx), std::move(cy), std::move(cz), std::move(jac));
            NW_PROFILER_POP();
            gpuPreprocessTimings_.hostCoordsDownloadSec = std::chrono::duration<double>(Clock::now() - tStage).count();
        }

        cudaFree(d_coordX); d_coordX = nullptr;
        cudaFree(d_coordY); d_coordY = nullptr;
        cudaFree(d_coordZ); d_coordZ = nullptr;
        cudaFree(d_gll_z);  d_gll_z = nullptr;

        // ------------------------------------------------------------------
        // Allocate UPML Auxiliary State & LSRK45 Register Buffers
        // Note: Surface scratch buffers (d_fEN_, d_fHN_, d_flux_) and UPML
        // residual buffer (d_resPmlAux_) are eliminated because those
        // intermediate quantities are kept in GPU registers by the fused
        // multi-operation kernels.
        // ------------------------------------------------------------------
        if (maxPml_ > 0) {
            NW_GPU_CHECK(cudaMalloc(&d_pmlAux_, 6 * static_cast<size_t>(nptsPml_) * sizeof(double)));
            NW_GPU_CHECK(cudaMemsetAsync(d_pmlAux_, 0, 6 * static_cast<size_t>(nptsPml_) * sizeof(double), stream_));

            NW_GPU_CHECK(cudaMalloc(&d_kPmlAux_, 6 * static_cast<size_t>(nptsPml_) * sizeof(double)));
            NW_GPU_CHECK(cudaMemsetAsync(d_kPmlAux_, 0, 6 * static_cast<size_t>(nptsPml_) * sizeof(double), stream_));
        }

        NW_GPU_CHECK(cudaStreamSynchronize(stream_));
        bInitialized_ = true;
        return true;
    }

    void setIncidentWave(const IncidentPlaneWaveConfig& incWave) {
        incWave_ = incWave;
        incWave_.updateMagneticPolarization();
    }

    void uploadState(const double* hostState, size_t size) {
        assert(bInitialized_ && size == static_cast<size_t>(totalEntries_));
        NW_PROFILER_PUSH("DgSolver::uploadState");
        if (numHaloPoints_ == 0) {
            NW_GPU_CHECK(cudaMemcpyAsync(d_state_, hostState, size * sizeof(double), cudaMemcpyHostToDevice, stream_));
        } else {
            NW_GPU_CHECK(cudaMemcpy2DAsync(
                d_state_, stateStride_ * sizeof(double),
                hostState, npts_ * sizeof(double),
                npts_ * sizeof(double), 6,
                cudaMemcpyHostToDevice, stream_
            ));
        }
        NW_GPU_CHECK(cudaStreamSynchronize(stream_));
        NW_PROFILER_POP();
    }

    void downloadState(double* hostState, size_t size) const {
        assert(bInitialized_ && size == static_cast<size_t>(totalEntries_));
        NW_PROFILER_PUSH("DgSolver::downloadState");
        if (numHaloPoints_ == 0) {
            NW_GPU_CHECK(cudaMemcpyAsync(hostState, d_state_, size * sizeof(double), cudaMemcpyDeviceToHost, stream_));
        } else {
            NW_GPU_CHECK(cudaMemcpy2DAsync(
                hostState, npts_ * sizeof(double),
                d_state_, stateStride_ * sizeof(double),
                npts_ * sizeof(double), 6,
                cudaMemcpyDeviceToHost, stream_
            ));
        }
        NW_GPU_CHECK(cudaStreamSynchronize(stream_));
        NW_PROFILER_POP();
    }

    /**
     * @brief Advances the complete 3D Maxwell system by dt using 5-stage LSRK45
     *        with fused multi-operation GPU kernels.
     *
     * Semi-discrete Maxwell curl equations:
     *      dE/dt = +(1/eps) * curl(H) - (1/eps) * M^{-1} [ F^*_E ]
     *      dH/dt = -(1/mu)  * curl(E) - (1/mu)  * M^{-1} [ F^*_H ]
     *
     * In each LSRK45 stage, 4 fused multi-operation GPU kernels are launched:
     *   1) Fused 2-Operation Volume Curl (`gpu_volume_curl_kernel`):
     *        - Operation 1: Volume Curl of H for Electric residual (+curl(H) -> res_E)
     *        - Operation 2: Volume Curl of E for Magnetic residual (-curl(E) -> res_H)
     *        (Overlapped with non-blocking MPI_Irecv / MPI_Isend halo exchange when active)
     *   2) Fused 3-Operation Surface Flux Interface (`gpu_surface_flux_kernel`):
     *        - Operation 1: Face Trace Restriction (gather interior E^-, H^- in registers)
     *        - Operation 2: Upwind/Central Riemann Numerical Flux & Boundary Conditions (F^*_E, F^*_H)
     *        - Operation 3: Surface Flux Lift / Scatter into Volume Residual (rhs += dA * F^*)
     *   3) Fused 2-Operation UPML Step (`gpu_pml_step_kernel`, when maxPml_ > 0):
     *        - Operation 1: UPML Auxiliary Differential Equation (ADE) Coupling (res_DN, res_BN, rhs_E, rhs_H)
     *        - Operation 2: Auxiliary Inverse Mass Scaling & LSRK45 Stage Update for (DN, BN)
     *   4) Fused 2-Operation Inverse Mass + LSRK45 Stage Update (`gpu_inv_mass_lsrk45_update_kernel`):
     *        - Operation 1: Diagonal Inverse Mass Matrix Scaling (r_scaled = rhs / (J * w_ijk))
     *        - Operation 2: LSRK45 Stage Update (k = a_s * k + dt * r_scaled, state = state + b_s * k)
     *
     * Zero host-device transfers and zero intermediate scratch buffers occur during this method.
     */
    void step(double dt, double time) {
        assert(bInitialized_);
        NW_PROFILER_PUSH("DgSolver::step");

        // Pointers into the 6 field components of state_ (using stateStride_)
        const double* Ex = d_state_ + 0 * stateStride_;
        const double* Ey = d_state_ + 1 * stateStride_;
        const double* Ez = d_state_ + 2 * stateStride_;
        const double* Hx = d_state_ + 3 * stateStride_;
        const double* Hy = d_state_ + 4 * stateStride_;
        const double* Hz = d_state_ + 5 * stateStride_;

        // Pointers into the 6 field components of rhs_ (local volume points only)
        double* resEx = d_rhs_ + 0 * npts_;
        double* resEy = d_rhs_ + 1 * npts_;
        double* resEz = d_rhs_ + 2 * npts_;
        double* resHx = d_rhs_ + 3 * npts_;
        double* resHy = d_rhs_ + 4 * npts_;
        double* resHz = d_rhs_ + 5 * npts_;

        // Advance through the 5 stages of Low-Storage Runge-Kutta (LSRK45)
        for (int stage = 0; stage < 5; ++stage) {
            const double stageTime = time + rk4c_[stage] * dt;
            NW_PROFILER_PUSH("LSRK45_Stage");
#ifdef NEKWAVE_ENABLE_MPI
            std::vector<MPI_Request> mpiReqs;
            if (numHaloPoints_ > 0) {
                NW_PROFILER_PUSH("MPI_Irecv");
                mpiReqs.reserve(2 * haloExchanges_.size());
                // 1. Post non-blocking receives for ghost traces
                for (size_t h = 0; h < haloExchanges_.size(); ++h) {
                    const auto& ex = haloExchanges_[h];
                    int tag = 2000 + stage;
                    MPI_Request req;
                    MPI_Irecv(d_mpiRecvBuf_ + exchangeBufOffsets_[h], 6 * ex.numPoints, MPI_DOUBLE,
                              ex.neighborRank, tag, MPI_COMM_WORLD, &req);
                    mpiReqs.push_back(req);
                }
                NW_PROFILER_POP();

                NW_PROFILER_PUSH("MPI_PackHalo");
                // 2. Pack send buffer on GPU from local volume traces
                nekwave::device::launch_pack_halo(
                    d_state_, d_mpiSendBuf_, d_sendVolIndices_,
                    d_sendBufOffsets_, d_exchangeSizes_,
                    numHaloPoints_, stateStride_, stream_
                );
                NW_GPU_CHECK(cudaStreamSynchronize(stream_));
                NW_PROFILER_POP();

                NW_PROFILER_PUSH("MPI_Isend");
                // 3. Post non-blocking sends directly from GPU memory (CUDA/ROCm-aware MPI)
                for (size_t h = 0; h < haloExchanges_.size(); ++h) {
                    const auto& ex = haloExchanges_[h];
                    int tag = 2000 + stage;
                    MPI_Request req;
                    MPI_Isend(d_mpiSendBuf_ + exchangeBufOffsets_[h], 6 * ex.numPoints, MPI_DOUBLE,
                              ex.neighborRank, tag, MPI_COMM_WORLD, &req);
                    mpiReqs.push_back(req);
                }
                NW_PROFILER_POP();
            }
#endif

            NW_PROFILER_PUSH("Launch_VolumeCurl_Fused2Op");
            // ------------------------------------------------------------------
            // Kernel 1 (Fused 2-Operation Volume Curl):
            //   - Operation 1: Volume Curl of H for Electric residual (+curl(H) -> resEx..resEz)
            //   - Operation 2: Volume Curl of E for Magnetic residual (-curl(E) -> resHx..resHz)
            // Loads curvilinear metric factors (rx..tz, jac) once per stage.
            // ------------------------------------------------------------------
            nekwave::device::launch_volume_curl(
                Ex, Ey, Ez, Hx, Hy, Hz,
                resEx, resEy, resEz, resHx, resHy, resHz,
                d_D_, d_w3_, d_jac_,
                d_rx_, d_sx_, d_tx_,
                d_ry_, d_sy_, d_ty_,
                d_rz_, d_sz_, d_tz_,
                N_, nelt_, stream_
            );
            NW_PROFILER_POP();

#ifdef NEKWAVE_ENABLE_MPI
            // Synchronize MPI communication before surface flux evaluation
            if (numHaloPoints_ > 0) {
                NW_PROFILER_PUSH("MPI_Waitall");
                MPI_Waitall(static_cast<int>(mpiReqs.size()), mpiReqs.data(), MPI_STATUSES_IGNORE);
                NW_PROFILER_POP();

                NW_PROFILER_PUSH("MPI_UnpackHalo");
                // Unpack received ghost data into ghost slots of d_state_
                nekwave::device::launch_unpack_halo(
                    d_mpiRecvBuf_, d_state_, d_recvBufOffsets_,
                    d_exchangeSizes_, numHaloPoints_, npts_, stateStride_, stream_
                );
                NW_PROFILER_POP();
            }
#endif

            NW_PROFILER_PUSH("Launch_SurfaceFlux_Fused3Op");
            // ------------------------------------------------------------------
            // Kernel 2 (Fused 3-Operation Surface Interface Kernel):
            //   - Operation 1: Face Trace Restriction (gather interior E^-, H^- in registers)
            //   - Operation 2: Upwind/Central Riemann Numerical Flux & BCs (F^*_E, F^*_H in registers)
            //   - Operation 3: Surface Flux Lift / Scatter into Volume Residual (rhs += dA * F^*)
            // ------------------------------------------------------------------
            nekwave::device::launch_surface_flux(
                d_state_, d_volIdxMinus_, d_volIdxPlus_, d_bcType_,
                d_nx_, d_ny_, d_nz_, d_dA_, d_rhs_,
                c0_, N_, totalFacePoints_, npts_, stateStride_, stream_,
                d_fx_, d_fy_, d_fz_, incWave_, stageTime
            );
            NW_PROFILER_POP();

            // ------------------------------------------------------------------
            // Kernel 3 (Fused 2-Operation UPML Step):
            //   - Operation 1: UPML Auxiliary Differential Equation (ADE) Coupling
            //   - Operation 2: Auxiliary Inverse Mass Scaling & LSRK45 Stage Update for (DN, BN)
            // ------------------------------------------------------------------
            if (maxPml_ > 0) {
                NW_PROFILER_PUSH("Launch_PmlStep_Fused2Op");
                nekwave::device::launch_pml_step(
                    d_state_, d_rhs_, d_pmlAux_, d_kPmlAux_,
                    d_pmlPtr_, d_pmlSigma_, d_jac_, d_w3_,
                    rk4a_[stage], rk4b_[stage], dt,
                    maxPml_, nxyz_, npts_, stateStride_, stream_
                );
                NW_PROFILER_POP();
            }

            NW_PROFILER_PUSH("Launch_InvMass_LSRK45_Fused2Op");
            // ------------------------------------------------------------------
            // Kernel 4 (Fused 2-Operation Inverse Mass + LSRK45 Stage Update):
            //   - Operation 1: Diagonal Inverse Mass Matrix Scaling (r_scaled = rhs / (J * w_ijk))
            //   - Operation 2: LSRK45 Stage Update (k = a_s * k + dt * r_scaled, state += b_s * k)
            // ------------------------------------------------------------------
            nekwave::device::launch_inv_mass_lsrk45_update(
                d_state_, d_k_, d_rhs_, d_jac_, d_w3_,
                rk4a_[stage], rk4b_[stage], dt,
                nelt_, nxyz_, npts_, stateStride_, stream_
            );
            NW_PROFILER_POP();

            NW_PROFILER_POP(); // LSRK45_Stage
        }

        // Stream synchronization only occurs at the end of the full time step
        NW_PROFILER_PUSH("StreamSynchronize");
        NW_GPU_CHECK(cudaStreamSynchronize(stream_));
        NW_PROFILER_POP();

        NW_PROFILER_POP(); // DgSolver::step
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
        cudaFree(d_bcType_); d_bcType_ = nullptr;
        cudaFree(d_nx_); d_nx_ = nullptr;
        cudaFree(d_ny_); d_ny_ = nullptr;
        cudaFree(d_nz_); d_nz_ = nullptr;
        cudaFree(d_dA_); d_dA_ = nullptr;
        if (d_fx_) { cudaFree(d_fx_); d_fx_ = nullptr; }
        if (d_fy_) { cudaFree(d_fy_); d_fy_ = nullptr; }
        if (d_fz_) { cudaFree(d_fz_); d_fz_ = nullptr; }

        if (d_pmlPtr_) { cudaFree(d_pmlPtr_); d_pmlPtr_ = nullptr; }
        if (d_pmlSigma_) { cudaFree(d_pmlSigma_); d_pmlSigma_ = nullptr; }
        if (d_pmlAux_) { cudaFree(d_pmlAux_); d_pmlAux_ = nullptr; }
        if (d_kPmlAux_) { cudaFree(d_kPmlAux_); d_kPmlAux_ = nullptr; }

        if (d_sendVolIndices_) { cudaFree(d_sendVolIndices_); d_sendVolIndices_ = nullptr; }
        if (d_sendBufOffsets_) { cudaFree(d_sendBufOffsets_); d_sendBufOffsets_ = nullptr; }
        if (d_recvBufOffsets_) { cudaFree(d_recvBufOffsets_); d_recvBufOffsets_ = nullptr; }
        if (d_exchangeSizes_) { cudaFree(d_exchangeSizes_); d_exchangeSizes_ = nullptr; }
        if (d_mpiSendBuf_) { cudaFree(d_mpiSendBuf_); d_mpiSendBuf_ = nullptr; }
        if (d_mpiRecvBuf_) { cudaFree(d_mpiRecvBuf_); d_mpiRecvBuf_ = nullptr; }

        numHaloPoints_ = 0;
        stateStride_ = 0;
        maxPml_ = 0;
        nptsPml_ = 0;
        haloExchanges_.clear();
        exchangeBufOffsets_.clear();

        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }

        bInitialized_ = false;
    }

    MemoryAndHardwareStats getMemoryAndHardwareStats() const {
        MemoryAndHardwareStats stats;
        stats.orderN = N_;
        stats.numElements = nelt_;
        stats.npts = npts_;
        stats.totalFacePoints = totalFacePoints_;
        stats.maxPmlElements = maxPml_;
        stats.nptsPml = nptsPml_;
        stats.numHaloPoints = numHaloPoints_;

        stats.stateBuffersBytes = (6 * static_cast<size_t>(stateStride_) + 2 * static_cast<size_t>(totalEntries_)) * sizeof(double);
        stats.metricBuffersBytes = (static_cast<size_t>(N_ * N_) + static_cast<size_t>(nxyz_) + 10 * static_cast<size_t>(npts_)) * sizeof(double);
        stats.faceTopologyBytes = 3 * static_cast<size_t>(totalFacePoints_) * sizeof(int) +
                                  7 * static_cast<size_t>(totalFacePoints_) * sizeof(double);
        stats.faceScratchBytes = 0;
        if (maxPml_ > 0) {
            stats.pmlBuffersBytes = static_cast<size_t>(maxPml_) * sizeof(int) +
                                    (3 * static_cast<size_t>(npts_) + 12 * static_cast<size_t>(nptsPml_)) * sizeof(double);
        }
        if (numHaloPoints_ > 0) {
            stats.mpiHaloBuffersBytes = 4 * static_cast<size_t>(numHaloPoints_) * sizeof(int) +
                                        12 * static_cast<size_t>(numHaloPoints_) * sizeof(double);
        }
        stats.totalNekWaveVramBytes = stats.stateBuffersBytes + stats.metricBuffersBytes +
                                      stats.faceTopologyBytes + stats.faceScratchBytes +
                                      stats.pmlBuffersBytes + stats.mpiHaloBuffersBytes;

        int dev = 0;
#if defined(USE_HIP) || defined(NEKWAVE_ENABLE_HIP) || defined(__HIPCC__) || defined(__HIP__)
        if (hipGetDevice(&dev) == hipSuccess) {
            hipDeviceProp_t prop;
            if (hipGetDeviceProperties(&prop, dev) == hipSuccess) {
                stats.deviceName = prop.name;
                stats.archName = prop.gcnArchName;
                stats.computeUnits = prop.multiProcessorCount;
                stats.clockRateKHz = prop.clockRate;
                stats.memoryClockRateKHz = prop.memoryClockRate;
                stats.memoryBusWidthBits = prop.memoryBusWidth;
                stats.totalGlobalMemBytes = prop.totalGlobalMem;
                stats.sharedMemPerBlockBytes = prop.sharedMemPerBlock;
            }
        }
        size_t freeMem = 0, totalMem = 0;
        if (hipMemGetInfo(&freeMem, &totalMem) == hipSuccess) {
            stats.driverFreeVramBytes = freeMem;
            stats.driverUsedVramBytes = (totalMem >= freeMem) ? (totalMem - freeMem) : 0;
            if (stats.totalGlobalMemBytes == 0) stats.totalGlobalMemBytes = totalMem;
        }
#else
        if (cudaGetDevice(&dev) == cudaSuccess) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
                stats.deviceName = prop.name;
                stats.archName = "sm_" + std::to_string(prop.major * 10 + prop.minor);
                stats.computeUnits = prop.multiProcessorCount;
                stats.clockRateKHz = prop.clockRate;
                stats.memoryClockRateKHz = prop.memoryClockRate;
                stats.memoryBusWidthBits = prop.memoryBusWidth;
                stats.totalGlobalMemBytes = prop.totalGlobalMem;
                stats.sharedMemPerBlockBytes = prop.sharedMemPerBlock;
            }
        }
        size_t freeMem = 0, totalMem = 0;
        if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess) {
            stats.driverFreeVramBytes = freeMem;
            stats.driverUsedVramBytes = (totalMem >= freeMem) ? (totalMem - freeMem) : 0;
            if (stats.totalGlobalMemBytes == 0) stats.totalGlobalMemBytes = totalMem;
        }
#endif
        return stats;
    }

    GpuPreprocessTimings getGpuPreprocessTimings() const {
        return gpuPreprocessTimings_;
    }

private:
    bool bInitialized_;
    cudaStream_t stream_;
    GpuPreprocessTimings gpuPreprocessTimings_;

    int N_ = 0;
    int nelt_ = 0;
    int nxyz_ = 0;
    int npts_ = 0;
    int numHaloPoints_ = 0;
    int stateStride_ = 0;
    int totalEntries_ = 0;
    int totalFacePoints_ = 0;
    int maxPml_ = 0;
    int nptsPml_ = 0;
    double c0_ = 1.0;
    IncidentPlaneWaveConfig incWave_;

    std::vector<double> rk4a_;
    std::vector<double> rk4b_;
    std::vector<double> rk4c_;

    std::vector<MpiHaloExchangeInfo> haloExchanges_;
    std::vector<int> exchangeBufOffsets_;

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
    int* d_bcType_ = nullptr;
    double* d_nx_ = nullptr;
    double* d_ny_ = nullptr;
    double* d_nz_ = nullptr;
    double* d_dA_ = nullptr;
    double* d_fx_ = nullptr;
    double* d_fy_ = nullptr;
    double* d_fz_ = nullptr;

    // UPML device buffers (only allocated when maxPml_ > 0)
    int* d_pmlPtr_ = nullptr;
    double* d_pmlSigma_ = nullptr;
    double* d_pmlAux_ = nullptr;
    double* d_kPmlAux_ = nullptr;

    // MPI Halo exchange device buffers
    int* d_sendVolIndices_ = nullptr;
    int* d_sendBufOffsets_ = nullptr;
    int* d_recvBufOffsets_ = nullptr;
    int* d_exchangeSizes_ = nullptr;
    double* d_mpiSendBuf_ = nullptr;
    double* d_mpiRecvBuf_ = nullptr;
};

// ==============================================================================
// DgSolver Public Facade Implementation
// ==============================================================================

DgSolver::DgSolver() : impl_(new Impl()) {}

DgSolver::~DgSolver() = default;

bool DgSolver::initialize(const Mesh& mesh, double c0) {
    return impl_->initialize(mesh, c0);
}

bool DgSolver::preprocessAndInitialize(Mesh& mesh, double c0,
                                       double* outDxmin,
                                       bool downloadHostCoords) {
    return impl_->preprocessAndInitialize(mesh, c0, outDxmin, downloadHostCoords);
}

DgSolver::GpuPreprocessTimings DgSolver::getGpuPreprocessTimings() const {
    return impl_->getGpuPreprocessTimings();
}

void DgSolver::setIncidentWave(const IncidentPlaneWaveConfig& incWave) {
    impl_->setIncidentWave(incWave);
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

DgSolver::MemoryAndHardwareStats DgSolver::getMemoryAndHardwareStats() const {
    return impl_->getMemoryAndHardwareStats();
}
