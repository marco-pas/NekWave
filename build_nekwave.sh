#!/bin/bash
# ==============================================================================
# NekWave Build Script for PDC KTH (NVIDIA CUDA / A100 + Parallel HDF5 + MPI)
# Usage: ./build_nekwave.sh [Debug|Release|RelWithDebInfo]
# ==============================================================================
set -e

BUILD_TYPE="${1:-Debug}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=================================================================="
echo " Building NekWave on PDC KTH (NVIDIA A100 / CUDA + MPI + HDF5)"
echo " Build Type   : ${BUILD_TYPE}"
echo " Source Dir   : ${SCRIPT_DIR}"
echo " Build Dir    : ${BUILD_DIR}"
echo "=================================================================="

# 1. Load PDC environment
if ! type module &>/dev/null; then
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    fi
fi

if [ -f "${SCRIPT_DIR}/config.sh" ]; then
    echo "--> Loading environment from config.sh..."
    source "${SCRIPT_DIR}/config.sh"
elif [ -f "${SCRIPT_DIR}/env_pdc.sh" ]; then
    echo "--> Loading environment from env_pdc.sh..."
    source "${SCRIPT_DIR}/env_pdc.sh"
else
    echo "--> Loading default modules for PDC KTH..."
    module load openmpi/4.1.6-gcc-12.2.0-kvteryp 2>/dev/null || module load openmpi/4.1.4-gcc-12.2.0-77fifwj 2>/dev/null || true
    module load hdf5/1.14.3-openmpi-4.1.6-gcc-12.2.0-2z7sqsi 2>/dev/null || module load hdf5/1.12.2-openmpi-4.1.4-gcc-11.3.0-2m5andk 2>/dev/null || true
    export LD_LIBRARY_PATH="/lib64:${LD_LIBRARY_PATH}"
fi

# 2. Configure CMake
# Defaults to NVIDIA CUDA sm_80 for this machine's dual A100 GPUs.
# Supports AMD ROCm/HIP if ENABLE_HIP=ON is explicitly provided.
if [ "${ENABLE_HIP}" = "ON" ] || [ "${ENABLE_HIP}" = "1" ]; then
    echo "--> Configuring with AMD ROCm / HIP backend (arch: ${CMAKE_HIP_ARCHITECTURES:-gfx90a})..."
    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
        -DENABLE_HIP=ON \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES:-gfx90a}" \
        -DENABLE_MPI=ON \
        -DBUILD_TESTING=ON
else
    echo "--> Configuring with NVIDIA CUDA backend (arch: ${CMAKE_CUDA_ARCHITECTURES:-80})..."
    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
        -DENABLE_HIP=OFF \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-80}" \
        -DENABLE_MPI=ON \
        -DBUILD_TESTING=ON
fi

# 3. Build executable and test targets
NUM_JOBS="${NPROC:-8}"
echo "--> Compiling with -j ${NUM_JOBS}..."
cmake --build "${BUILD_DIR}" -j "${NUM_JOBS}"

echo "=================================================================="
echo " NekWave built successfully!"
echo " Binary: ${BUILD_DIR}/nekwave"
echo "=================================================================="
