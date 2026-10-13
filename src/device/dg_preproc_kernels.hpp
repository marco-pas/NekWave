#ifndef NW_DEVICE_DG_PREPROC_KERNELS_HPP
#define NW_DEVICE_DG_PREPROC_KERNELS_HPP

#include <cstddef>
#include "gpu_runtime.hpp"
#include "boundary_conditions.hpp"

namespace nekwave {
namespace device {

/**
 * @brief Computes 3D GLL physical coordinates, Jacobian determinants, and
 *        contravariant metric tensors (rx..tz) directly on the GPU from the 8 element corners.
 */
void launch_compute_volume_metrics(
    const double* d_corners, const double* d_gll_z,
    double* d_coordX, double* d_coordY, double* d_coordZ,
    double* d_jac,
    double* d_rx, double* d_sx, double* d_tx,
    double* d_ry, double* d_sy, double* d_ty,
    double* d_rz, double* d_sz, double* d_tz,
    int N, int nelt, int npts,
    cudaStream_t stream = nullptr
);

/**
 * @brief Computes face outward unit normals (nx, ny, nz), surface Jacobian dA,
 *        face node coordinates (fx, fy, fz), and D4 neighbor volume index matching
 *        (volIdxMinus, volIdxPlus, bcType) directly on the GPU.
 */
void launch_setup_face_data(
    const int* d_faceElem, const int* d_faceId,
    const int* d_neighborElem, const int* d_neighborFace,
    const int* d_faceBcCode,
    const double* d_gll_w,
    const double* d_coordX, const double* d_coordY, const double* d_coordZ,
    const double* d_jac,
    const double* d_rx, const double* d_sx, const double* d_tx,
    const double* d_ry, const double* d_sy, const double* d_ty,
    const double* d_rz, const double* d_sz, const double* d_tz,
    int* d_volIdxMinus, int* d_volIdxPlus, int* d_bcType,
    double* d_nx, double* d_ny, double* d_nz, double* d_dA,
    double* d_fx, double* d_fy, double* d_fz,
    int N, int numFaces, int totalFacePoints,
    cudaStream_t stream = nullptr,
    const int* d_mpiCutGhostIdx = nullptr
);

/**
 * @brief Computes UPML polynomial damping conductivity profiles (sigma_x, sigma_y, sigma_z)
 *        directly in GPU VRAM for all PML elements.
 */
void launch_compute_pml_sigma(
    const int* d_pmlPtr, const int* d_pmlTag,
    const double* d_coordX, const double* d_coordY, const double* d_coordZ,
    const double pmlInner[6], const double pmlOuter[6],
    double pmlOrder, double pmlReflectErr,
    double* d_pmlSigma,
    int maxPml, int nxyz, int npts,
    cudaStream_t stream = nullptr
);

/**
 * @brief Computes the minimum physical distance between adjacent GLL collocation nodes
 *        (and curvilinear metric spacing) across all elements on the GPU.
 */
double launch_compute_min_node_distance(
    const double* d_coordX, const double* d_coordY, const double* d_coordZ,
    const double* d_rx, const double* d_sx, const double* d_tx,
    const double* d_ry, const double* d_sy, const double* d_ty,
    const double* d_rz, const double* d_sz, const double* d_tz,
    double dxiMin, int N, int nelt, int npts,
    cudaStream_t stream = nullptr
);

} // namespace device
} // namespace nekwave

#endif // NW_DEVICE_DG_PREPROC_KERNELS_HPP

