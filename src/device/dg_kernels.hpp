#ifndef NW_DEVICE_DG_KERNELS_HPP
#define NW_DEVICE_DG_KERNELS_HPP

#include <cstddef>
#include <cuda_runtime.h>

/**
 * @file dg_kernels.hpp
 * @brief Clean C++ launcher interface for NekWave DG-SEM device kernels.
 *
 * Exposes dispatch functions that configure launch grids, thread blocks,
 * and shared memory for GPU kernels without leaking CUDA execution syntax (<<<...>>>)
 * into high-level solver code.
 */

// Error-checking helper for CUDA API invocations
#ifndef NW_GPU_CHECK
#define NW_GPU_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "[NekWave GPU ERROR] %s at %s:%d\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)
#endif

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
    const int* d_volIdxMinus, int totalFacePoints, int npts,
    cudaStream_t stream = nullptr
);

/**
 * @brief Launches numerical surface flux evaluation kernel (Central flux + PEC).
 */
void launch_compute_flux(
    const double* d_state, const double* d_fEN, const double* d_fHN,
    const int* d_volIdxPlus, const int* d_isPEC,
    const double* d_nx, const double* d_ny, const double* d_nz,
    double* d_flux, double c0, int totalFacePoints, int npts,
    cudaStream_t stream = nullptr
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
    double rk4a, double rk4b, double dt, int totalEntries,
    cudaStream_t stream = nullptr
);

} // namespace device
} // namespace nekwave

#endif // NW_DEVICE_DG_KERNELS_HPP

