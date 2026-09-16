#include "mxm_cuda.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <cuda_runtime.h>

// CPU column-major reference matching Physics::mxm
void cpu_mxm(const double* A, int n1, const double* B, int n2, double* C, int n3) {
    for (int j = 0; j < n3; ++j) {
        for (int i = 0; i < n1; ++i) {
            double sum = 0.0;
            for (int l = 0; l < n2; ++l) {
                sum += A[i + l * n1] * B[l + j * n2];
            }
            C[i + j * n1] = sum;
        }
    }
}

// Helper to fill matrix with pseudo-random smooth test values
void fill_matrix(std::vector<double>& mat, int rows, int cols, double seed_offset) {
    mat.resize(rows * cols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            mat[i + j * rows] = std::sin(0.13 * (i + 1) + 0.29 * (j + 1) + seed_offset);
        }
    }
}

// Test case verification function
bool verify_shape(int n1, int n2, int n3, const std::string& label) {
    std::vector<double> A, B, C_cpu, C_gpu;
    fill_matrix(A, n1, n2, 1.0);
    fill_matrix(B, n2, n3, 2.0);
    C_cpu.assign(n1 * n3, 0.0);
    C_gpu.assign(n1 * n3, 0.0);

    cpu_mxm(A.data(), n1, B.data(), n2, C_cpu.data(), n3);
    nw_cuda_mxm(A.data(), n1, B.data(), n2, C_gpu.data(), n3);

    double maxDiff = 0.0;
    for (size_t i = 0; i < C_cpu.size(); ++i) {
        double diff = std::abs(C_cpu[i] - C_gpu[i]);
        if (diff > maxDiff) maxDiff = diff;
    }

    bool bPassed = (maxDiff < 1e-12);
    std::cout << "  [" << (bPassed ? "PASS" : "FAIL") << "] " 
              << std::left << std::setw(30) << label 
              << " (" << n1 << "x" << n2 << " * " << n2 << "x" << n3 << " -> " << n1 << "x" << n3 << ")"
              << " Max Diff: " << std::scientific << std::setprecision(3) << maxDiff
              << std::defaultfloat << std::endl;
    return bPassed;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "        NekWave CUDA Matrix Multiplication Verification    " << std::endl;
    std::cout << "==========================================================" << std::endl;

    if (!nw_cuda_is_available()) {
        std::cerr << "ERROR: No CUDA-capable GPU found on system!" << std::endl;
        return 1;
    }

    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Using GPU Device " << device << ": " << prop.name
              << " (Compute Capability " << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n" << std::endl;

    nw_cuda_init(65536);

    int totalTests = 0;
    int passedTests = 0;

    std::cout << "--- 1. Verification of Tensor Contraction Shapes ---" << std::endl;
    // Polynomial orders N = 4, 6, 8, 10
    for (int N : {4, 6, 8, 10}) {
        std::string tag = "N=" + std::to_string(N);
        // Contraction 1: ur = (I (x) I (x) D) u -> (N x N) * (N x N^2)
        totalTests++;
        if (verify_shape(N, N, N * N, tag + " ur contraction")) passedTests++;

        // Contraction 2: us_k = u_k * D^T -> (N x N) * (N x N)
        totalTests++;
        if (verify_shape(N, N, N, tag + " us slice contraction")) passedTests++;

        // Contraction 3: ut = u * D^T -> (N^2 x N) * (N x N)
        totalTests++;
        if (verify_shape(N * N, N, N, tag + " ut contraction")) passedTests++;
    }

    // Additional general dimensions and larger tile-straddling shapes
    totalTests++;
    if (verify_shape(32, 32, 32, "32x32 square tiled")) passedTests++;
    totalTests++;
    if (verify_shape(48, 64, 48, "48x64x48 non-symmetric tiled")) passedTests++;
    totalTests++;
    if (verify_shape(7, 13, 19, "Arbitrary primes (7x13 * 13x19)")) passedTests++;

    std::cout << "\nVerification Summary: " << passedTests << " / " << totalTests << " passed.\n" << std::endl;

    // --- 2. Performance Benchmark ---
    std::cout << "--- 2. Performance Benchmarks (N=6, DG-SEM Polynomial Order 5) ---" << std::endl;
    const int N = 6;
    const int n1 = N, n2 = N, n3 = N * N; // Shape of ur contraction: (6x6) * (6x36)
    const int iters = 50000;

    std::vector<double> A, B, C_cpu, C_gpu;
    fill_matrix(A, n1, n2, 1.0);
    fill_matrix(B, n2, n3, 2.0);
    C_cpu.assign(n1 * n3, 0.0);
    C_gpu.assign(n1 * n3, 0.0);

    // Warmup
    nw_cuda_mxm(A.data(), n1, B.data(), n2, C_gpu.data(), n3);

    // Benchmark CPU
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < iters; ++k) {
        cpu_mxm(A.data(), n1, B.data(), n2, C_cpu.data(), n3);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Benchmark GPU Host-Pointer API (includes PCIe copy + kernel + sync)
    t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < iters; ++k) {
        nw_cuda_mxm(A.data(), n1, B.data(), n2, C_gpu.data(), n3);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double gpu_host_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Benchmark Raw GPU Device Kernel (zero PCIe transfers)
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n1 * n2 * sizeof(double));
    cudaMalloc(&d_B, n2 * n3 * sizeof(double));
    cudaMalloc(&d_C, n1 * n3 * sizeof(double));
    cudaMemcpy(d_A, A.data(), n1 * n2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), n2 * n3 * sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < iters; ++k) {
        nw_cuda_mxm_device(d_A, n1, d_B, n2, d_C, n3, nullptr);
    }
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double gpu_dev_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    double totalFlops = 2.0 * n1 * n2 * n3 * iters;
    double cpu_gflops = (totalFlops / (cpu_time_ms * 1e-3)) * 1e-9;
    double gpu_host_gflops = (totalFlops / (gpu_host_time_ms * 1e-3)) * 1e-9;
    double gpu_dev_gflops = (totalFlops / (gpu_dev_time_ms * 1e-3)) * 1e-9;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  CPU Reference Time (" << iters << " iters):     " << cpu_time_ms << " ms  (" << cpu_gflops << " GFLOPS)" << std::endl;
    std::cout << "  GPU Host API Time  (incl PCIe copies): " << gpu_host_time_ms << " ms  (" << gpu_host_gflops << " GFLOPS)" << std::endl;
    std::cout << "  GPU Raw Device Kernel Time (zero copy):  " << gpu_dev_time_ms << " ms  (" << gpu_dev_gflops << " GFLOPS)" << std::endl;
    std::cout << "  Raw GPU Kernel Speedup over CPU:       " << (cpu_time_ms / gpu_dev_time_ms) << "x" << std::endl;
    std::cout << "==========================================================" << std::endl;

    nw_cuda_finalize();
    return (passedTests == totalTests) ? 0 : 1;
}

