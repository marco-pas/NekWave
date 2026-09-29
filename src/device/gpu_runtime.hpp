#ifndef NW_DEVICE_GPU_RUNTIME_HPP
#define NW_DEVICE_GPU_RUNTIME_HPP

/**
 * @file gpu_runtime.hpp
 * @brief Unified cross-platform abstraction layer for NVIDIA CUDA and AMD ROCm / HIP.
 *
 * Enables seamless portability across NVIDIA GPUs (via CUDA) and AMD GPUs
 * (via HIP on ROCm platforms like HPE Cray EX / AMD Instinct MI200/MI300 series).
 */

#include <cstdio>
#include <cstdlib>

#if defined(USE_HIP) || defined(NEKWAVE_ENABLE_HIP) || defined(__HIPCC__) || defined(__HIP__)

// ==============================================================================
// AMD ROCm / HIP Backend
// ==============================================================================
#include <hip/hip_runtime.h>

#if defined(__has_include)
#  if __has_include(<roctx.h>)
#    include <roctx.h>
#    define NW_PROFILER_PUSH(name) roctxRangePushA(name)
#    define NW_PROFILER_POP()      roctxRangePop()
#  endif
#endif

#ifndef NW_PROFILER_PUSH
#  define NW_PROFILER_PUSH(name) ((void)0)
#  define NW_PROFILER_POP()      ((void)0)
#endif

// Map CUDA runtime identifiers directly to HIP runtime functions and types
#ifndef cudaStream_t
#define cudaStream_t             hipStream_t
#endif
#ifndef cudaError_t
#define cudaError_t              hipError_t
#endif
#ifndef cudaSuccess
#define cudaSuccess              hipSuccess
#endif
#ifndef cudaGetErrorString
#define cudaGetErrorString       hipGetErrorString
#endif
#ifndef cudaMalloc
#define cudaMalloc               hipMalloc
#endif
#ifndef cudaFree
#define cudaFree                 hipFree
#endif
#ifndef cudaMemcpy
#define cudaMemcpy               hipMemcpy
#endif
#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync          hipMemcpyAsync
#endif
#ifndef cudaMemcpy2DAsync
#define cudaMemcpy2DAsync        hipMemcpy2DAsync
#endif
#ifndef cudaMemsetAsync
#define cudaMemsetAsync          hipMemsetAsync
#endif
#ifndef cudaStreamCreate
#define cudaStreamCreate         hipStreamCreate
#endif
#ifndef cudaStreamDestroy
#define cudaStreamDestroy        hipStreamDestroy
#endif
#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize    hipStreamSynchronize
#endif
#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#endif
#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#endif
#ifndef cudaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#endif

// Generic NekWave GPU runtime aliases
using gpuStream_t = hipStream_t;
using gpuError_t  = hipError_t;
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString

#ifndef NW_GPU_CHECK
#define NW_GPU_CHECK(call) do { \
    hipError_t _err = (call); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "[NekWave HIP ERROR] %s at %s:%d (code %d)\n", \
                     hipGetErrorString(_err), __FILE__, __LINE__, (int)_err); \
        std::abort(); \
    } \
} while(0)
#endif

#else

// ==============================================================================
// NVIDIA CUDA Backend
// ==============================================================================
#include <cuda_runtime.h>

#ifndef USE_HIP
#include <nvtx3/nvToolsExt.h>
#define NW_PROFILER_PUSH(name) nvtxRangePushA(name)
#define NW_PROFILER_POP()      nvtxRangePop()
#else
#define NW_PROFILER_PUSH(name) ((void)0)
#define NW_PROFILER_POP()      ((void)0)
#endif

// Generic NekWave GPU runtime aliases
using gpuStream_t = cudaStream_t;
using gpuError_t  = cudaError_t;
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString

#ifndef NW_GPU_CHECK
#define NW_GPU_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::fprintf(stderr, "[NekWave CUDA ERROR] %s at %s:%d (code %d)\n", \
                     cudaGetErrorString(_err), __FILE__, __LINE__, (int)_err); \
        std::abort(); \
    } \
} while(0)
#endif

#endif // USE_HIP

#endif // NW_DEVICE_GPU_RUNTIME_HPP
