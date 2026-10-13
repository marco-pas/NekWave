#include "dg_preproc_kernels.hpp"
#include "gpu_runtime.hpp"
#include "boundary_conditions.hpp"
#include <cmath>
#include <cstdio>

// ==============================================================================
// GPU-Native Mesh Preprocessing Kernels (Volume Metrics, Faces, UPML, CFL)
// ==============================================================================

/**
 * @brief Evaluates trilinear coordinate mapping, Jacobian determinant, and
 *        contravariant metric tensor J^{-T} at every 3D GLL collocation point
 *        directly on the GPU from the 8 element corners.
 */
__global__ void gpu_compute_volume_metrics_kernel(
    const double* __restrict__ d_corners,
    const double* __restrict__ d_gll_z,
    double* __restrict__ d_coordX,
    double* __restrict__ d_coordY,
    double* __restrict__ d_coordZ,
    double* __restrict__ d_jac,
    double* __restrict__ d_rx, double* __restrict__ d_sx, double* __restrict__ d_tx,
    double* __restrict__ d_ry, double* __restrict__ d_sy, double* __restrict__ d_ty,
    double* __restrict__ d_rz, double* __restrict__ d_sz, double* __restrict__ d_tz,
    int N, int nelt, int npts)
{
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ptIdx >= npts) return;

    int N2 = N * N;
    int N3 = N2 * N;
    int e = ptIdx / N3;
    if (e >= nelt) return;

    int m = ptIdx % N3;
    int i = m % N;
    int j = (m / N) % N;
    int k = m / N2;

    double r = d_gll_z[i];
    double s = d_gll_z[j];
    double t = d_gll_z[k];

    double N_fn[8];
    N_fn[0] = 0.125 * (1.0 - r) * (1.0 - s) * (1.0 - t);
    N_fn[1] = 0.125 * (1.0 + r) * (1.0 - s) * (1.0 - t);
    N_fn[2] = 0.125 * (1.0 + r) * (1.0 + s) * (1.0 - t);
    N_fn[3] = 0.125 * (1.0 - r) * (1.0 + s) * (1.0 - t);
    N_fn[4] = 0.125 * (1.0 - r) * (1.0 - s) * (1.0 + t);
    N_fn[5] = 0.125 * (1.0 + r) * (1.0 - s) * (1.0 + t);
    N_fn[6] = 0.125 * (1.0 + r) * (1.0 + s) * (1.0 + t);
    N_fn[7] = 0.125 * (1.0 - r) * (1.0 + s) * (1.0 + t);

    double dN_dr[8], dN_ds[8], dN_dt[8];
    dN_dr[0] = -0.125 * (1.0 - s) * (1.0 - t);
    dN_dr[1] =  0.125 * (1.0 - s) * (1.0 - t);
    dN_dr[2] =  0.125 * (1.0 + s) * (1.0 - t);
    dN_dr[3] = -0.125 * (1.0 + s) * (1.0 - t);
    dN_dr[4] = -0.125 * (1.0 - s) * (1.0 + t);
    dN_dr[5] =  0.125 * (1.0 - s) * (1.0 + t);
    dN_dr[6] =  0.125 * (1.0 + s) * (1.0 + t);
    dN_dr[7] = -0.125 * (1.0 + s) * (1.0 + t);

    dN_ds[0] = -0.125 * (1.0 - r) * (1.0 - t);
    dN_ds[1] = -0.125 * (1.0 + r) * (1.0 - t);
    dN_ds[2] =  0.125 * (1.0 + r) * (1.0 - t);
    dN_ds[3] =  0.125 * (1.0 - r) * (1.0 - t);
    dN_ds[4] = -0.125 * (1.0 - r) * (1.0 + t);
    dN_ds[5] = -0.125 * (1.0 + r) * (1.0 + t);
    dN_ds[6] =  0.125 * (1.0 + r) * (1.0 + t);
    dN_ds[7] =  0.125 * (1.0 - r) * (1.0 + t);

    dN_dt[0] = -0.125 * (1.0 - r) * (1.0 - s);
    dN_dt[1] = -0.125 * (1.0 + r) * (1.0 - s);
    dN_dt[2] = -0.125 * (1.0 + r) * (1.0 + s);
    dN_dt[3] = -0.125 * (1.0 - r) * (1.0 + s);
    dN_dt[4] =  0.125 * (1.0 - r) * (1.0 - s);
    dN_dt[5] =  0.125 * (1.0 + r) * (1.0 - s);
    dN_dt[6] =  0.125 * (1.0 + r) * (1.0 + s);
    dN_dt[7] =  0.125 * (1.0 - r) * (1.0 + s);

    const double* c = d_corners + static_cast<size_t>(e) * 24;
    double x = 0.0, y = 0.0, z = 0.0;
    double xr = 0.0, yr = 0.0, zr = 0.0;
    double xs = 0.0, ys = 0.0, zs = 0.0;
    double xt = 0.0, yt = 0.0, zt = 0.0;

    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        double vx = c[a * 3 + 0];
        double vy = c[a * 3 + 1];
        double vz = c[a * 3 + 2];

        x += N_fn[a] * vx;
        y += N_fn[a] * vy;
        z += N_fn[a] * vz;

        xr += dN_dr[a] * vx;
        yr += dN_dr[a] * vy;
        zr += dN_dr[a] * vz;

        xs += dN_ds[a] * vx;
        ys += dN_ds[a] * vy;
        zs += dN_ds[a] * vz;

        xt += dN_dt[a] * vx;
        yt += dN_dt[a] * vy;
        zt += dN_dt[a] * vz;
    }

    d_coordX[ptIdx] = x;
    d_coordY[ptIdx] = y;
    d_coordZ[ptIdx] = z;

    double detJ = xr * (ys * zt - yt * zs)
                - xs * (yr * zt - yt * zr)
                + xt * (yr * zs - ys * zr);
    d_jac[ptIdx] = detJ;

    double invJ = 1.0 / detJ;
    d_rx[ptIdx] = invJ * (ys * zt - yt * zs);
    d_sx[ptIdx] = invJ * (yt * zr - yr * zt);
    d_tx[ptIdx] = invJ * (yr * zs - ys * zr);

    d_ry[ptIdx] = invJ * (xt * zs - xs * zt);
    d_sy[ptIdx] = invJ * (xr * zt - xt * zr);
    d_ty[ptIdx] = invJ * (xs * zr - xr * zs);

    d_rz[ptIdx] = invJ * (xs * yt - xt * ys);
    d_sz[ptIdx] = invJ * (xt * yr - xr * yt);
    d_tz[ptIdx] = invJ * (xr * ys - xs * yr);
}

__device__ __forceinline__ int dev_face_node_vol_index(int elemId, int faceId, int p, int q, int N, int N3) {
    int i = 0, j = 0, k = 0;
    switch (faceId) {
        case 0: i = p;     j = 0;     k = q;     break;
        case 1: i = N - 1; j = p;     k = q;     break;
        case 2: i = p;     j = N - 1; k = q;     break;
        case 3: i = 0;     j = p;     k = q;     break;
        case 4: i = p;     j = q;     k = 0;     break;
        case 5: i = p;     j = q;     k = N - 1; break;
        default: break;
    }
    return elemId * N3 + i + N * (j + N * k);
}

/**
 * @brief Computes face outward normals, surface Jacobian dA, and D4 square
 *        orientation neighbor matching for all face quadrature points on the GPU.
 */
__global__ void gpu_setup_face_data_kernel(
    const int* __restrict__ d_faceElem,
    const int* __restrict__ d_faceId,
    const int* __restrict__ d_neighborElem,
    const int* __restrict__ d_neighborFace,
    const int* __restrict__ d_faceBcCode,
    const int* __restrict__ d_mpiCutGhostIdx,
    const double* __restrict__ d_gll_w,
    const double* __restrict__ d_coordX,
    const double* __restrict__ d_coordY,
    const double* __restrict__ d_coordZ,
    const double* __restrict__ d_jac,
    const double* __restrict__ d_rx, const double* __restrict__ d_sx, const double* __restrict__ d_tx,
    const double* __restrict__ d_ry, const double* __restrict__ d_sy, const double* __restrict__ d_ty,
    const double* __restrict__ d_rz, const double* __restrict__ d_sz, const double* __restrict__ d_tz,
    int* __restrict__ d_volIdxMinus,
    int* __restrict__ d_volIdxPlus,
    int* __restrict__ d_bcType,
    double* __restrict__ d_nx,
    double* __restrict__ d_ny,
    double* __restrict__ d_nz,
    double* __restrict__ d_dA,
    double* __restrict__ d_fx,
    double* __restrict__ d_fy,
    double* __restrict__ d_fz,
    int N, int numFaces, int totalFacePoints)
{
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ptIdx >= totalFacePoints) return;

    int N2 = N * N;
    int N3 = N2 * N;
    int fIdx = ptIdx / N2;
    if (fIdx >= numFaces) return;

    int localPt = ptIdx % N2;
    int pA = localPt % N;
    int qA = localPt / N;

    int eA = d_faceElem[fIdx];
    int fA = d_faceId[fIdx];
    int eB = d_neighborElem[fIdx];
    int fB = d_neighborFace[fIdx];
    BcType bc = static_cast<BcType>(d_faceBcCode[fIdx]);

    bool isMpiCut = (bc == BcType::MPI_CUT);
    bool isExterior = (!isMpiCut && (bc == BcType::PEC || bc == BcType::PMC || bc == BcType::PML || eB < 0));
    bool isPeriodic = (!isExterior && !isMpiCut && bc == BcType::PERIODIC);

    int vM = dev_face_node_vol_index(eA, fA, pA, qA, N, N3);
    d_volIdxMinus[ptIdx] = vM;

    double J = d_jac[vM];
    double w2 = d_gll_w[pA] * d_gll_w[qA];

    double Nx = 0.0, Ny = 0.0, Nz = 0.0;
    switch (fA) {
        case 0: Nx = -J * d_sx[vM]; Ny = -J * d_sy[vM]; Nz = -J * d_sz[vM]; break;
        case 1: Nx = +J * d_rx[vM]; Ny = +J * d_ry[vM]; Nz = +J * d_rz[vM]; break;
        case 2: Nx = +J * d_sx[vM]; Ny = +J * d_sy[vM]; Nz = +J * d_sz[vM]; break;
        case 3: Nx = -J * d_rx[vM]; Ny = -J * d_ry[vM]; Nz = -J * d_rz[vM]; break;
        case 4: Nx = -J * d_tx[vM]; Ny = -J * d_ty[vM]; Nz = -J * d_tz[vM]; break;
        case 5: Nx = +J * d_tx[vM]; Ny = +J * d_ty[vM]; Nz = +J * d_tz[vM]; break;
        default: break;
    }

    double L = sqrt(Nx * Nx + Ny * Ny + Nz * Nz);
    d_nx[ptIdx] = (L > 1.0e-14) ? (Nx / L) : 0.0;
    d_ny[ptIdx] = (L > 1.0e-14) ? (Ny / L) : 0.0;
    d_nz[ptIdx] = (L > 1.0e-14) ? (Nz / L) : 0.0;
    d_dA[ptIdx] = L * w2;

    double xA = d_coordX[vM];
    double yA = d_coordY[vM];
    double zA = d_coordZ[vM];
    d_fx[ptIdx] = xA;
    d_fy[ptIdx] = yA;
    d_fz[ptIdx] = zA;

    int vP = -1;
    if (isMpiCut) {
        vP = (d_mpiCutGhostIdx != nullptr) ? d_mpiCutGhostIdx[ptIdx] : -1;
    } else if (isExterior) {
        vP = -1;
    } else if (isPeriodic) {
        vP = dev_face_node_vol_index(eB, fB, pA, qA, N, N3);
    } else {
        // Test the 8 D4 square symmetries on the face corners first (O(1))
        if (N > 1) {
            int vA00 = dev_face_node_vol_index(eA, fA, 0, 0, N, N3);
            int vA10 = dev_face_node_vol_index(eA, fA, N - 1, 0, N, N3);
            int vA01 = dev_face_node_vol_index(eA, fA, 0, N - 1, N, N3);
            double xA00 = d_coordX[vA00], yA00 = d_coordY[vA00], zA00 = d_coordZ[vA00];
            double xA10 = d_coordX[vA10], yA10 = d_coordY[vA10], zA10 = d_coordZ[vA10];
            double xA01 = d_coordX[vA01], yA01 = d_coordY[vA01], zA01 = d_coordZ[vA01];

            for (int orient = 0; orient < 8; ++orient) {
                int pB00 = (orient & 1) ? (N - 1) : 0;
                int qB00 = (orient & 2) ? (N - 1) : 0;
                int vB00 = dev_face_node_vol_index(eB, fB, pB00, qB00, N, N3);
                double dx00 = xA00 - d_coordX[vB00];
                double dy00 = yA00 - d_coordY[vB00];
                double dz00 = zA00 - d_coordZ[vB00];
                if (dx00 * dx00 + dy00 * dy00 + dz00 * dz00 > 1.0e-12) continue;

                int u10 = (orient & 4) ? 0 : (N - 1);
                int v10 = (orient & 4) ? (N - 1) : 0;
                int pB10 = (orient & 1) ? (N - 1 - u10) : u10;
                int qB10 = (orient & 2) ? (N - 1 - v10) : v10;
                int vB10 = dev_face_node_vol_index(eB, fB, pB10, qB10, N, N3);
                double dx10 = xA10 - d_coordX[vB10];
                double dy10 = yA10 - d_coordY[vB10];
                double dz10 = zA10 - d_coordZ[vB10];
                if (dx10 * dx10 + dy10 * dy10 + dz10 * dz10 > 1.0e-12) continue;

                int u01 = (orient & 4) ? (N - 1) : 0;
                int v01 = (orient & 4) ? 0 : (N - 1);
                int pB01 = (orient & 1) ? (N - 1 - u01) : u01;
                int qB01 = (orient & 2) ? (N - 1 - v01) : v01;
                int vB01 = dev_face_node_vol_index(eB, fB, pB01, qB01, N, N3);
                double dx01 = xA01 - d_coordX[vB01];
                double dy01 = yA01 - d_coordY[vB01];
                double dz01 = zA01 - d_coordZ[vB01];
                if (dx01 * dx01 + dy01 * dy01 + dz01 * dz01 > 1.0e-12) continue;

                int u = (orient & 4) ? qA : pA;
                int v = (orient & 4) ? pA : qA;
                int pB = (orient & 1) ? (N - 1 - u) : u;
                int qB = (orient & 2) ? (N - 1 - v) : v;
                int candVB = dev_face_node_vol_index(eB, fB, pB, qB, N, N3);
                double dx = xA - d_coordX[candVB];
                double dy = yA - d_coordY[candVB];
                double dz = zA - d_coordZ[candVB];
                if (dx * dx + dy * dy + dz * dz < 1.0e-12) {
                    vP = candVB;
                }
                break;
            }
        }

        if (vP < 0) {
            double bestDist2 = 1.0e30;
            for (int qB = 0; qB < N; ++qB) {
                for (int pB = 0; pB < N; ++pB) {
                    int candVB = dev_face_node_vol_index(eB, fB, pB, qB, N, N3);
                    double dx = xA - d_coordX[candVB];
                    double dy = yA - d_coordY[candVB];
                    double dz = zA - d_coordZ[candVB];
                    double d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 < bestDist2) {
                        bestDist2 = d2;
                        vP = candVB;
                    }
                }
            }
        }
    }

    d_volIdxPlus[ptIdx] = vP;
    BcType ptBc = bc;
    if (vP < 0 && (ptBc == BcType::INTERIOR || ptBc == BcType::PERIODIC || ptBc == BcType::MPI_CUT)) {
        ptBc = BcType::PEC;
    }
    d_bcType[ptIdx] = static_cast<int>(ptBc);
}

struct PmlExtremaParams {
    double inner[6];
    double outer[6];
};

/**
 * @brief Evaluates UPML polynomial damping conductivity profiles directly in GPU VRAM.
 */
__global__ void gpu_compute_pml_sigma_kernel(
    const int* __restrict__ d_pmlPtr,
    const int* __restrict__ d_pmlTag,
    const double* __restrict__ d_coordX,
    const double* __restrict__ d_coordY,
    const double* __restrict__ d_coordZ,
    PmlExtremaParams extents,
    double pmlOrder,
    double pmlReflectErr,
    double* __restrict__ d_pmlSigma,
    int maxPml, int nxyz, int npts)
{
    int pmlPtIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pmlPtIdx >= maxPml * nxyz) return;

    int e = pmlPtIdx / nxyz;
    int i = pmlPtIdx % nxyz;
    int ie = d_pmlPtr[e];
    int tag = d_pmlTag[e];
    int j = ie * nxyz + i;

    double x = d_coordX[j];
    double y = d_coordY[j];
    double z = d_coordZ[j];
    const double eta = 1.0;

    #pragma unroll
    for (int face = 0; face < 6; ++face) {
        if ((tag & (1 << face)) != 0) {
            int axis = face / 2;
            double inner = extents.inner[face];
            double outer = extents.outer[face];
            double width = fabs(outer - inner);
            if (width > 1.0e-14) {
                double sigmaMax = -(pmlOrder + 1.0) * log(pmlReflectErr) / (2.0 * eta * width);
                double coord = (axis == 0) ? x : ((axis == 1) ? y : z);
                double zero2one = (coord - inner) / (outer - inner);
                zero2one = fmax(0.0, fmin(1.0, zero2one));
                d_pmlSigma[axis * npts + j] = sigmaMax * pow(zero2one, pmlOrder);
            }
        }
    }
}

/**
 * @brief Computes per-block minimum squared GLL node distance across all elements on the GPU.
 */
__global__ void gpu_compute_min_node_distance_kernel(
    const double* __restrict__ d_coordX,
    const double* __restrict__ d_coordY,
    const double* __restrict__ d_coordZ,
    const double* __restrict__ d_rx, const double* __restrict__ d_sx, const double* __restrict__ d_tx,
    const double* __restrict__ d_ry, const double* __restrict__ d_sy, const double* __restrict__ d_ty,
    const double* __restrict__ d_rz, const double* __restrict__ d_sz, const double* __restrict__ d_tz,
    double dxiMin2, int N, int npts,
    double* __restrict__ d_blockMinD2)
{
    __shared__ double s_min[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double localMinD2 = 1.0e20;
    if (idx < npts) {
        int N2 = N * N;
        int N3 = N2 * N;
        int e = idx / N3;
        int m = idx % N3;
        int i = m % N;
        int j = (m / N) % N;
        int k = m / N2;
        int offset = e * N3;

        double x0 = d_coordX[idx];
        double y0 = d_coordY[idx];
        double z0 = d_coordZ[idx];

        if (i + 1 < N) {
            int idx_r = offset + (i + 1) + N * (j + N * k);
            double dx = d_coordX[idx_r] - x0;
            double dy = d_coordY[idx_r] - y0;
            double dz = d_coordZ[idx_r] - z0;
            localMinD2 = fmin(localMinD2, dx * dx + dy * dy + dz * dz);
        }
        if (j + 1 < N) {
            int idx_s = offset + i + N * ((j + 1) + N * k);
            double dx = d_coordX[idx_s] - x0;
            double dy = d_coordY[idx_s] - y0;
            double dz = d_coordZ[idx_s] - z0;
            localMinD2 = fmin(localMinD2, dx * dx + dy * dy + dz * dz);
        }
        if (k + 1 < N) {
            int idx_t = offset + i + N * (j + N * (k + 1));
            double dx = d_coordX[idx_t] - x0;
            double dy = d_coordY[idx_t] - y0;
            double dz = d_coordZ[idx_t] - z0;
            localMinD2 = fmin(localMinD2, dx * dx + dy * dy + dz * dz);
        }

        double g2 = (d_rx[idx] * d_rx[idx] + d_ry[idx] * d_ry[idx] + d_rz[idx] * d_rz[idx])
                  + (d_sx[idx] * d_sx[idx] + d_sy[idx] * d_sy[idx] + d_sz[idx] * d_sz[idx])
                  + (d_tx[idx] * d_tx[idx] + d_ty[idx] * d_ty[idx] + d_tz[idx] * d_tz[idx]);
        if (g2 > 1.0e-20) {
            localMinD2 = fmin(localMinD2, (3.0 * dxiMin2) / g2);
        }
    }

    s_min[tid] = localMinD2;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[tid] = fmin(s_min[tid], s_min[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_blockMinD2[blockIdx.x] = s_min[0];
    }
}

__global__ void gpu_reduce_min_d2_kernel(
    const double* __restrict__ d_blockMinD2,
    int numBlocks,
    double* __restrict__ d_outMinD2)
{
    __shared__ double s_min[256];
    int tid = threadIdx.x;
    double localMin = 1.0e20;
    for (int b = tid; b < numBlocks; b += blockDim.x) {
        localMin = fmin(localMin, d_blockMinD2[b]);
    }
    s_min[tid] = localMin;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[tid] = fmin(s_min[tid], s_min[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_outMinD2[0] = s_min[0];
    }
}

// ==============================================================================
// Launcher Dispatch Wrappers (Exposed C++ API)
// ==============================================================================

namespace nekwave {
namespace device {

void launch_compute_volume_metrics(
    const double* d_corners, const double* d_gll_z,
    double* d_coordX, double* d_coordY, double* d_coordZ,
    double* d_jac,
    double* d_rx, double* d_sx, double* d_tx,
    double* d_ry, double* d_sy, double* d_ty,
    double* d_rz, double* d_sz, double* d_tz,
    int N, int nelt, int npts,
    cudaStream_t stream)
{
    if (npts <= 0) return;
    int blockSize = 256;
    int numBlocks = (npts + blockSize - 1) / blockSize;

    gpu_compute_volume_metrics_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_corners, d_gll_z,
        d_coordX, d_coordY, d_coordZ,
        d_jac,
        d_rx, d_sx, d_tx,
        d_ry, d_sy, d_ty,
        d_rz, d_sz, d_tz,
        N, nelt, npts
    );
    NW_GPU_CHECK(cudaGetLastError());
}

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
    cudaStream_t stream,
    const int* d_mpiCutGhostIdx)
{
    if (totalFacePoints <= 0) return;
    int blockSize = 256;
    int numBlocks = (totalFacePoints + blockSize - 1) / blockSize;

    gpu_setup_face_data_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_faceElem, d_faceId, d_neighborElem, d_neighborFace, d_faceBcCode,
        d_mpiCutGhostIdx,
        d_gll_w,
        d_coordX, d_coordY, d_coordZ,
        d_jac,
        d_rx, d_sx, d_tx,
        d_ry, d_sy, d_ty,
        d_rz, d_sz, d_tz,
        d_volIdxMinus, d_volIdxPlus, d_bcType,
        d_nx, d_ny, d_nz, d_dA,
        d_fx, d_fy, d_fz,
        N, numFaces, totalFacePoints
    );
    NW_GPU_CHECK(cudaGetLastError());
}

void launch_compute_pml_sigma(
    const int* d_pmlPtr, const int* d_pmlTag,
    const double* d_coordX, const double* d_coordY, const double* d_coordZ,
    const double pmlInner[6], const double pmlOuter[6],
    double pmlOrder, double pmlReflectErr,
    double* d_pmlSigma,
    int maxPml, int nxyz, int npts,
    cudaStream_t stream)
{
    if (maxPml <= 0 || npts <= 0) return;
    PmlExtremaParams extents;
    for (int f = 0; f < 6; ++f) {
        extents.inner[f] = pmlInner[f];
        extents.outer[f] = pmlOuter[f];
    }
    int nptsPml = maxPml * nxyz;
    int blockSize = 256;
    int numBlocks = (nptsPml + blockSize - 1) / blockSize;

    gpu_compute_pml_sigma_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_pmlPtr, d_pmlTag,
        d_coordX, d_coordY, d_coordZ,
        extents, pmlOrder, pmlReflectErr,
        d_pmlSigma, maxPml, nxyz, npts
    );
    NW_GPU_CHECK(cudaGetLastError());
}

double launch_compute_min_node_distance(
    const double* d_coordX, const double* d_coordY, const double* d_coordZ,
    const double* d_rx, const double* d_sx, const double* d_tx,
    const double* d_ry, const double* d_sy, const double* d_ty,
    const double* d_rz, const double* d_sz, const double* d_tz,
    double dxiMin, int N, int nelt, int npts,
    cudaStream_t stream)
{
    (void)nelt;
    if (npts <= 0) return 1.0;

    int blockSize = 256;
    int numBlocks = (npts + blockSize - 1) / blockSize;
    double dxiMin2 = dxiMin * dxiMin;

    double* d_blockMinD2 = nullptr;
    double* d_outMinD2 = nullptr;
    NW_GPU_CHECK(cudaMalloc(&d_blockMinD2, static_cast<size_t>(numBlocks) * sizeof(double)));
    NW_GPU_CHECK(cudaMalloc(&d_outMinD2, sizeof(double)));

    gpu_compute_min_node_distance_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_coordX, d_coordY, d_coordZ,
        d_rx, d_sx, d_tx,
        d_ry, d_sy, d_ty,
        d_rz, d_sz, d_tz,
        dxiMin2, N, npts,
        d_blockMinD2
    );
    NW_GPU_CHECK(cudaGetLastError());

    gpu_reduce_min_d2_kernel<<<1, 256, 0, stream>>>(
        d_blockMinD2, numBlocks, d_outMinD2
    );
    NW_GPU_CHECK(cudaGetLastError());

    double h_minD2 = 1.0;
    NW_GPU_CHECK(cudaMemcpyAsync(&h_minD2, d_outMinD2, sizeof(double), cudaMemcpyDeviceToHost, stream));
    NW_GPU_CHECK(cudaStreamSynchronize(stream));

    NW_GPU_CHECK(cudaFree(d_blockMinD2));
    NW_GPU_CHECK(cudaFree(d_outMinD2));

    return (h_minD2 > 0.0) ? std::sqrt(h_minD2) : 1.0e-10;
}

} // namespace device
} // namespace nekwave

