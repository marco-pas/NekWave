#ifndef NW_DEVICE_DG_SIM_KERNELS_HPP
#define NW_DEVICE_DG_SIM_KERNELS_HPP

#include <cstddef>
#include "gpu_runtime.hpp"
#include "boundary_conditions.hpp"

namespace nekwave {
namespace device {

/**
 * @brief Launches fused dual-field 3D volume curl kernel:
 *        Operation 1: Volume curl of H for Electric field residual (+nabla x H -> resE)
 *        Operation 2: Volume curl of E for Magnetic field residual (-nabla x E -> resH)
 *        Loads geometric metric factors (rx..tz, J) once per stage for both curls.
 */
void launch_volume_curl(
    const double* Ex, const double* Ey, const double* Ez,
    const double* Hx, const double* Hy, const double* Hz,
    double* resEx, double* resEy, double* resEz,
    double* resHx, double* resHy, double* resHz,
    const double* D, const double* w3_gll, const double* jac,
    const double* rx, const double* sx, const double* tx,
    const double* ry, const double* sy, const double* ty,
    const double* rz, const double* sz, const double* tz,
    int N, int nelt, cudaStream_t stream = nullptr
);

/**
 * @brief Launches fused 3-stage surface interface kernel:
 *        Operation 1 (Restrict): Gather interior volume traces (E^-, H^-) at volIdxMinus
 *        Operation 2 (Riemann Flux): Evaluate exterior trace jumps ([[E]], [[H]]), BCs,
 *                                    and Upwind/Central numerical flux (F^*_E, F^*_H)
 *        Operation 3 (Surface Lift): Scale by surface area dA and lift directly into d_rhs
 */
void launch_surface_flux(
    const double* d_state,
    const int* d_volIdxMinus, const int* d_volIdxPlus, const int* d_bcType,
    const double* d_nx, const double* d_ny, const double* d_nz, const double* d_dA,
    double* d_rhs, double c0, int N, int totalFacePoints, int npts, int stateStride,
    cudaStream_t stream = nullptr,
    const double* d_fx = nullptr, const double* d_fy = nullptr, const double* d_fz = nullptr,
    IncidentPlaneWaveConfig incWave = IncidentPlaneWaveConfig(), double stageTime = 0.0
);

/**
 * @brief Launches fused 2-stage UPML absorbing boundary kernel:
 *        Operation 1 (UPML ADE): Evaluate Auxiliary Differential Equations and update unscaled E/H rhs
 *        Operation 2 (Aux LSRK45): Scale auxiliary residuals by M^{-1} and advance pmlAux & kPmlAux
 */
void launch_pml_step(
    const double* d_state, double* d_rhs,
    double* d_pmlAux, double* d_kPmlAux,
    const int* d_pmlPtr, const double* d_pmlSigma,
    const double* d_jac, const double* d_w3,
    double rk4a, double rk4b, double dt,
    int maxPml, int nxyz, int npts, int stateStride,
    cudaStream_t stream = nullptr
);

/**
 * @brief Launches fused 2-stage inverse mass scaling and LSRK45 time integration kernel:
 *        Operation 1 (Inverse Mass): Scale volume residual by diagonal M^{-1} = 1 / (J * w_ijk)
 *        Operation 2 (LSRK45 Update): Advance k = a_s * k + dt * (M^{-1} rhs) and state += b_s * k
 */
void launch_inv_mass_lsrk45_update(
    double* d_state, double* d_k, const double* d_rhs,
    const double* d_jac, const double* d_w3,
    double rk4a, double rk4b, double dt,
    int nelt, int nxyz, int npts, int stateStride,
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

#endif // NW_DEVICE_DG_SIM_KERNELS_HPP

