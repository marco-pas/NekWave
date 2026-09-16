#ifndef NW_CUDA_MXM_CUDA_HPP
#define NW_CUDA_MXM_CUDA_HPP

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs column-major matrix multiplication C = A * B on the GPU using host pointers.
 *
 * Internally manages device transfers using reusable persistent buffers to avoid
 * allocation overhead on repeated calls.
 *
 * Matrix layouts (column-major):
 *   A is n1 x n2 (leading dimension n1): A[i + l * n1]
 *   B is n2 x n3 (leading dimension n2): B[l + j * n2]
 *   C is n1 x n3 (leading dimension n1): C[i + j * n1]
 *
 * Operation:
 *   C(i, j) = sum_{l=0}^{n2-1} A(i, l) * B(l, j)
 *
 * @param A  Pointer to host memory matrix A (n1 x n2)
 * @param n1 Number of rows in A and C
 * @param B  Pointer to host memory matrix B (n2 x n3)
 * @param n2 Number of columns in A and rows in B (inner contraction dimension)
 * @param C  Pointer to host memory matrix C (n1 x n3)
 * @param n3 Number of columns in B and C
 */
void nw_cuda_mxm(const double* A, int n1, const double* B, int n2, double* C, int n3);

/**
 * @brief Performs column-major matrix multiplication C = A * B directly on device pointers.
 *
 * This function assumes d_A, d_B, and d_C reside on the GPU device memory.
 * Zero host-to-device or device-to-host transfers are performed.
 *
 * @param d_A    Pointer to device memory matrix A (n1 x n2)
 * @param n1     Number of rows in A and C
 * @param d_B    Pointer to device memory matrix B (n2 x n3)
 * @param n2     Number of columns in A and rows in B
 * @param d_C    Pointer to device memory matrix C (n1 x n3)
 * @param n3     Number of columns in B and C
 * @param stream Optional CUDA stream pointer (cast to void*). If nullptr, default stream 0 is used.
 */
void nw_cuda_mxm_device(const double* d_A, int n1, const double* d_B, int n2, double* d_C, int n3, void* stream);

/**
 * @brief Checks if a compatible CUDA GPU is available on the host system.
 *
 * @return true if at least one CUDA-capable device is detected, false otherwise.
 */
bool nw_cuda_is_available(void);

/**
 * @brief Initializes CUDA context and pre-allocates scratch buffers for mxm operations.
 *
 * @param initial_capacity_doubles Optional initial scratch buffer capacity in doubles (default 65536).
 */
void nw_cuda_init(size_t initial_capacity_doubles);

/**
 * @brief Releases any persistent device scratch buffers allocated by the CUDA mxm subsystem.
 */
void nw_cuda_finalize(void);

#ifdef __cplusplus
}
#endif

#endif // NW_CUDA_MXM_CUDA_HPP

