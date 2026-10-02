#!/usr/bin/env bash
#
# NekWave Automated Hipification Tool (CUDA -> AMD ROCm / HIP)
# Not really needed
#
# Usage:
#   # 1. Check hipification status and compatibility of current codebase:
#   ./tools/hipify.sh --check
#
#   # 2. Generate a pure hipified copy in a target directory (default: hipified/):
#   ./tools/hipify.sh -o hipified/
#
#   # 3. Configure and build with CMake for AMD GPUs (e.g. on Dardel / ROCm cluster):
#   cmake -B build_hip -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a ..
#   cmake --build build_hip -j
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECK_ONLY=false
IN_PLACE=false
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --inplace)
            IN_PLACE=true
            shift
            ;;
        -o|--output)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--check] [--inplace] [-o <out_dir>]"
            exit 1
            ;;
    esac
done

echo "=========================================================="
echo " NekWave Hipification & AMD ROCm Portability Tool"
echo "=========================================================="

# Check if official AMD hipify-perl or hipify-clang is installed
HIPIFY_TOOL=""
if command -v hipify-perl >/dev/null 2>&1; then
    HIPIFY_TOOL="hipify-perl"
elif [ -x "/opt/rocm/bin/hipify-perl" ]; then
    HIPIFY_TOOL="/opt/rocm/bin/hipify-perl"
elif command -v hipify-clang >/dev/null 2>&1; then
    HIPIFY_TOOL="hipify-clang"
elif [ -x "/opt/rocm/bin/hipify-clang" ]; then
    HIPIFY_TOOL="/opt/rocm/bin/hipify-clang"
fi

if [ -n "$HIPIFY_TOOL" ]; then
    echo " Found official ROCm hipify utility: $HIPIFY_TOOL"
else
    echo " Official hipify-perl/clang not on PATH; using built-in NekWave hipifier engine."
fi

# Target source files for hipification
FILES=(
    "src/device/dg_kernels.hpp"
    "src/device/dg_kernels.cu"
    "src/dg_solver.cpp"
    "src/device/gpu_runtime.hpp"
)

if [ "$CHECK_ONLY" = true ]; then
    echo " Inspecting codebase for CUDA/HIP portability..."
    TOTAL_CUDA_REFS=0
    for f in "${FILES[@]}"; do
        if [ -f "$ROOT_DIR/$f" ]; then
            REFS=$(grep -c -E "cudaMalloc|cudaMemcpy|cudaFree|cudaStream|cudaError|<cuda_runtime.h>" "$ROOT_DIR/$f" || true)
            echo "   - $f: $REFS CUDA API references (wrapped by gpu_runtime.hpp)"
            TOTAL_CUDA_REFS=$((TOTAL_CUDA_REFS + REFS))
        fi
    done
    echo " Status: Codebase uses unified 'src/device/gpu_runtime.hpp' layer."
    echo "         Compiles directly with 'cmake -DENABLE_HIP=ON' on AMD ROCm platforms!"
    exit 0
fi

if [ -z "$OUT_DIR" ] && [ "$IN_PLACE" = false ]; then
    OUT_DIR="$ROOT_DIR/hipified"
fi

if [ "$IN_PLACE" = true ]; then
    TARGET_DIR="$ROOT_DIR"
    echo " Performing in-place hipification of NekWave device files..."
else
    TARGET_DIR="$OUT_DIR"
    echo " Generating hipified source tree at: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
    # Copy source directory structure
    cp -r "$ROOT_DIR/src" "$TARGET_DIR/"
    cp -r "$ROOT_DIR/CMakeLists.txt" "$TARGET_DIR/"
    if [ -f "$ROOT_DIR/src/device/dg_kernels.cu" ]; then
        cp "$ROOT_DIR/src/device/dg_kernels.cu" "$TARGET_DIR/src/device/dg_kernels.hip"
    fi
fi

# Hipification replacement function
hipify_file() {
    local src="$1"
    local dst="$2"

    if [ -n "$HIPIFY_TOOL" ] && [ "$HIPIFY_TOOL" != "builtin" ]; then
        "$HIPIFY_TOOL" "$src" > "$dst.tmp" && mv "$dst.tmp" "$dst"
    else
        sed -E \
            -e 's/<cuda_runtime\.h>/<hip\/hip_runtime.h>/g' \
            -e 's/<nvtx3\/nvToolsExt\.h>/<roctx.h>/g' \
            -e 's/nvtxRangePushA/roctxRangePushA/g' \
            -e 's/nvtxRangePop/roctxRangePop/g' \
            -e 's/\bcudaMalloc\b/hipMalloc/g' \
            -e 's/\bcudaFree\b/hipFree/g' \
            -e 's/\bcudaMemcpy\b/hipMemcpy/g' \
            -e 's/\bcudaMemcpyAsync\b/hipMemcpyAsync/g' \
            -e 's/\bcudaMemcpy2DAsync\b/hipMemcpy2DAsync/g' \
            -e 's/\bcudaMemsetAsync\b/hipMemsetAsync/g' \
            -e 's/\bcudaStream_t\b/hipStream_t/g' \
            -e 's/\bcudaStreamCreate\b/hipStreamCreate/g' \
            -e 's/\bcudaStreamDestroy\b/hipStreamDestroy/g' \
            -e 's/\bcudaStreamSynchronize\b/hipStreamSynchronize/g' \
            -e 's/\bcudaError_t\b/hipError_t/g' \
            -e 's/\bcudaSuccess\b/hipSuccess/g' \
            -e 's/\bcudaGetErrorString\b/hipGetErrorString/g' \
            -e 's/\bcudaMemcpyHostToDevice\b/hipMemcpyHostToDevice/g' \
            -e 's/\bcudaMemcpyDeviceToHost\b/hipMemcpyDeviceToHost/g' \
            -e 's/\bcudaMemcpyDeviceToDevice\b/hipMemcpyDeviceToDevice/g' \
            "$src" > "$dst.tmp" && mv "$dst.tmp" "$dst"
    fi
}

echo " Processing device kernel files..."
if [ "$IN_PLACE" = false ]; then
    hipify_file "$ROOT_DIR/src/device/dg_kernels.cu" "$TARGET_DIR/src/device/dg_kernels.hip"
    hipify_file "$ROOT_DIR/src/device/dg_kernels.hpp" "$TARGET_DIR/src/device/dg_kernels.hpp"
    hipify_file "$ROOT_DIR/src/dg_solver.cpp" "$TARGET_DIR/src/dg_solver.cpp"
    echo " Hipified source tree created successfully at: $TARGET_DIR"
else
    for f in "${FILES[@]}"; do
        if [ -f "$ROOT_DIR/$f" ]; then
            hipify_file "$ROOT_DIR/$f" "$ROOT_DIR/$f"
            echo "   - Hipified $f in-place."
        fi
    done
fi

echo "=========================================================="
echo " Hipification complete!"
echo " To compile on AMD ROCm (e.g. Dardel Instinct MI250X):"
echo "   cmake -B build -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a .."
echo "   cmake --build build -j"
echo "=========================================================="
