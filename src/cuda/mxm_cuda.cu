#include "mxm_cuda.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cstdio>

#define TILE_DIM 16

// Error-checking macro for CUDA API calls
#define NW_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "[NekWave CUDA ERROR] %s at %s:%d\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

// ==============================================================================
// CUDA Kernels for Column-Major Matrix Multiplication: C = A * B
// ==============================================================================

/**
 * @brief Direct 2D grid matrix multiplication kernel for column-major layout.
 *
 * Optimized for small matrix sizes typical in DG-SEM tensor contractions (N <= 16).
 * Each thread (tx, ty) directly computes C[i + j * n1] via register accumulation.
 *
 * Memory layout (column-major):
 *   A(i, l) = A[i + l * n1]
 *   B(l, j) = B[l + j * n2]
 *   C(i, j) = C[i + j * n1]
 */
__global__ void nw_cuda_mxm_kernel_direct(const double* __restrict__ A, int n1,
                                          const double* __restrict__ B, int n2,
                                          double* __restrict__ C, int n3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index in C and A
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Column index in C and B

    if (i < n1 && j < n3) {
        double sum = 0.0;
        #pragma unroll 4
        for (int l = 0; l < n2; ++l) {
            sum += A[i + l * n1] * B[l + j * n2];
        }
        C[i + j * n1] = sum;
    }
}

/**
 * @brief Tiled shared-memory matrix multiplication kernel for column-major layout.
 *
 * Uses TILE_DIM x TILE_DIM shared memory blocks to reduce global memory bandwidth
 * for larger contractions (e.g. N^2 x N or high polynomial orders).
 */
__global__ void nw_cuda_mxm_kernel_tiled(const double* __restrict__ A, int n1,
                                         const double* __restrict__ B, int n2,
                                         double* __restrict__ C, int n3) {
    __shared__ double sA[TILE_DIM][TILE_DIM];
    __shared__ double sB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.x * TILE_DIM + tx; // Row in C and A
    int j = blockIdx.y * TILE_DIM + ty; // Column in C and B

    double sum = 0.0;
    int numTiles = (n2 + TILE_DIM - 1) / TILE_DIM;

    for (int m = 0; m < numTiles; ++m) {
        // Load tile from A into shared memory: sA[tx][ty] = A(i, m * TILE_DIM + ty)
        int colA = m * TILE_DIM + ty;
        if (i < n1 && colA < n2) {
            sA[tx][ty] = A[i + colA * n1];
        } else {
            sA[tx][ty] = 0.0;
        }

        // Load tile from B into shared memory: sB[tx][ty] = B(m * TILE_DIM + tx, j)
        int rowB = m * TILE_DIM + tx;
        if (rowB < n2 && j < n3) {
            sB[tx][ty] = B[rowB + j * n2];
        } else {
            sB[tx][ty] = 0.0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += sA[tx][k] * sB[k][ty];
        }

        __syncthreads();
    }

    if (i < n1 && j < n3) {
        C[i + j * n1] = sum;
    }
}

// ==============================================================================
// Internal Persistent Buffer Management
// ==============================================================================

namespace {
    struct ScratchState {
        double* d_A = nullptr;
        double* d_B = nullptr;
        double* d_C = nullptr;
        size_t cap_A = 0;
        size_t cap_B = 0;
        size_t cap_C = 0;
        cudaStream_t stream = nullptr;
        bool bInitialized = false;
    };

    static ScratchState g_scratch;

    void ensure_capacity(size_t need_A, size_t need_B, size_t need_C) {
        if (!g_scratch.bInitialized) {
            NW_CUDA_CHECK(cudaStreamCreate(&g_scratch.stream));
            g_scratch.bInitialized = true;
        }

        if (need_A > g_scratch.cap_A) {
            if (g_scratch.d_A) {
                NW_CUDA_CHECK(cudaFree(g_scratch.d_A));
            }
            g_scratch.cap_A = std::max(need_A, g_scratch.cap_A * 2 + 1024);
            NW_CUDA_CHECK(cudaMalloc(&g_scratch.d_A, g_scratch.cap_A * sizeof(double)));
        }

        if (need_B > g_scratch.cap_B) {
            if (g_scratch.d_B) {
                NW_CUDA_CHECK(cudaFree(g_scratch.d_B));
            }
            g_scratch.cap_B = std::max(need_B, g_scratch.cap_B * 2 + 1024);
            NW_CUDA_CHECK(cudaMalloc(&g_scratch.d_B, g_scratch.cap_B * sizeof(double)));
        }

        if (need_C > g_scratch.cap_C) {
            if (g_scratch.d_C) {
                NW_CUDA_CHECK(cudaFree(g_scratch.d_C));
            }
            g_scratch.cap_C = std::max(need_C, g_scratch.cap_C * 2 + 1024);
            NW_CUDA_CHECK(cudaMalloc(&g_scratch.d_C, g_scratch.cap_C * sizeof(double)));
        }
    }
} // anonymous namespace

// ==============================================================================
// Public API Implementation
// ==============================================================================

extern "C" {

bool nw_cuda_is_available(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

void nw_cuda_init(size_t initial_capacity_doubles) {
    if (initial_capacity_doubles == 0) {
        initial_capacity_doubles = 65536;
    }
    ensure_capacity(initial_capacity_doubles, initial_capacity_doubles, initial_capacity_doubles);
}

void nw_cuda_finalize(void) {
    if (g_scratch.d_A) {
        cudaFree(g_scratch.d_A);
        g_scratch.d_A = nullptr;
        g_scratch.cap_A = 0;
    }
    if (g_scratch.d_B) {
        cudaFree(g_scratch.d_B);
        g_scratch.d_B = nullptr;
        g_scratch.cap_B = 0;
    }
    if (g_scratch.d_C) {
        cudaFree(g_scratch.d_C);
        g_scratch.d_C = nullptr;
        g_scratch.cap_C = 0;
    }
    if (g_scratch.stream) {
        cudaStreamDestroy(g_scratch.stream);
        g_scratch.stream = nullptr;
    }
    g_scratch.bInitialized = false;
}

void nw_cuda_mxm_device(const double* d_A, int n1, const double* d_B, int n2, double* d_C, int n3, void* stream) {
    if (n1 <= 0 || n2 <= 0 || n3 <= 0) return;

    cudaStream_t custream = stream ? static_cast<cudaStream_t>(stream) : 0;

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((n1 + block.x - 1) / block.x, (n3 + block.y - 1) / block.y);

    // Use tiled kernel for larger problems; direct kernel for small DG tensor contractions
    if (n1 <= 16 && n2 <= 16 && n3 <= 16) {
        nw_cuda_mxm_kernel_direct<<<grid, block, 0, custream>>>(d_A, n1, d_B, n2, d_C, n3);
    } else {
        nw_cuda_mxm_kernel_tiled<<<grid, block, 0, custream>>>(d_A, n1, d_B, n2, d_C, n3);
    }
}

void nw_cuda_mxm(const double* A, int n1, const double* B, int n2, double* C, int n3) {
    if (n1 <= 0 || n2 <= 0 || n3 <= 0) return;

    size_t sz_A = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    size_t sz_B = static_cast<size_t>(n2) * static_cast<size_t>(n3);
    size_t sz_C = static_cast<size_t>(n1) * static_cast<size_t>(n3);

    ensure_capacity(sz_A, sz_B, sz_C);

    // Asynchronously transfer inputs to device
    NW_CUDA_CHECK(cudaMemcpyAsync(g_scratch.d_A, A, sz_A * sizeof(double),
                                  cudaMemcpyHostToDevice, g_scratch.stream));
    NW_CUDA_CHECK(cudaMemcpyAsync(g_scratch.d_B, B, sz_B * sizeof(double),
                                  cudaMemcpyHostToDevice, g_scratch.stream));

    // Launch matrix multiplication kernel
    nw_cuda_mxm_device(g_scratch.d_A, n1, g_scratch.d_B, n2, g_scratch.d_C, n3, g_scratch.stream);

    // Asynchronously transfer result back to host and synchronize stream
    NW_CUDA_CHECK(cudaMemcpyAsync(C, g_scratch.d_C, sz_C * sizeof(double),
                                  cudaMemcpyDeviceToHost, g_scratch.stream));
    NW_CUDA_CHECK(cudaStreamSynchronize(g_scratch.stream));
}

} // extern "C"

