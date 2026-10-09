#ifndef NW_DEVICE_DG_KERNELS_HPP
#define NW_DEVICE_DG_KERNELS_HPP

#include <cstddef>
#include "gpu_runtime.hpp"
#include "boundary_conditions.hpp"

namespace nekwave {
namespace device {

/**
 * @brief Launches fused 3D tensor contraction and weighted physical curl kernel.
 */
void launch_volume_curl(
    const double* u1, const double* u2, const double* u3,
    double* w1, double* w2, double* w3,
    const double* D, const double* w3_gll, const double* jac,
    const double* rx, const double* sx, const double* tx,
    const double* ry, const double* sy, const double* ty,
    const double* rz, const double* sz, const double* tz,
    double sign, int N, int nelt, cudaStream_t stream = nullptr
);

/**
 * @brief Launches kernel restricting volume solution fields to element face traces.
 */
void launch_restrict_faces(
    const double* d_state, double* d_fEN, double* d_fHN,
    const int* d_volIdxMinus, int totalFacePoints, int stateStride,
    cudaStream_t stream = nullptr
);

/**
 * @brief Launches numerical surface flux evaluation kernel (Central/Upwind flux + PEC/PMC/PML/Periodic BCs).
 */
void launch_compute_flux(
    const double* d_state, const double* d_fEN, const double* d_fHN,
    const int* d_volIdxPlus, const int* d_bcType,
    const double* d_nx, const double* d_ny, const double* d_nz,
    double* d_flux, double c0, int totalFacePoints, int stateStride,
    cudaStream_t stream = nullptr,
    const double* d_fx = nullptr, const double* d_fy = nullptr, const double* d_fz = nullptr,
    IncidentPlaneWaveConfig incWave = IncidentPlaneWaveConfig(), double stageTime = 0.0
);

/**
 * @brief Launches kernel lifting numerical surface fluxes into volume residuals.
 */
void launch_add_flux(
    const double* d_flux, const int* d_volIdxMinus, const double* d_dA,
    double* d_rhs, int totalFacePoints, int npts,
    cudaStream_t stream = nullptr
);

/**
 * @brief Launches UPML Auxiliary Differential Equations (ADE) volume step kernel (matching NekCEM's pml_step).
 */
void launch_pml_step(
    const double* d_state, double* d_rhs,
    const double* d_pmlAux, double* d_resPmlAux,
    const int* d_pmlPtr, const double* d_pmlSigma,
    const double* d_jac, const double* d_w3,
    int maxPml, int nxyz, int npts, int stateStride,
    cudaStream_t stream = nullptr
);

/**
 * @brief Launches diagonal inverse mass matrix scaling kernel.
 */
void launch_inv_mass(
    double* d_rhs, const double* d_jac, const double* d_w3,
    int nelt, int nxyz, int npts,
    cudaStream_t stream = nullptr
);

/**
 * @brief Launches Low-Storage Runge-Kutta 4(5) stage update kernel.
 */
void launch_lsrk45_update(
    double* d_state, double* d_k, const double* d_rhs,
    double rk4a, double rk4b, double dt, int npts, int stateStride,
    cudaStream_t stream = nullptr
);

/**
 * @brief Packs cut face trace values from d_state into contiguous MPI send buffer.
 */
void launch_pack_halo(
    const double* d_state, double* d_sendBuf,
    const int* d_sendVolIndices, const int* d_sendBufOffsets,
    const int* d_exchangeSizes, int numHaloPoints, int stateStride,
    cudaStream_t stream = nullptr
);

/**
 * @brief Unpacks received MPI buffer into the ghost region of d_state.
 */
void launch_unpack_halo(
    const double* d_recvBuf, double* d_state,
    const int* d_recvBufOffsets, const int* d_exchangeSizes,
    int numHaloPoints, int npts, int stateStride,
    cudaStream_t stream = nullptr
);

} // namespace device
} // namespace nekwave

#endif // NW_DEVICE_DG_KERNELS_HPP

