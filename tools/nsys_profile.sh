#!/usr/bin/env bash
#
# NekWave Nsight Systems Profiling Helper
#
# Supports both serial and MPI multi-GPU profiling:
#
# Usage:
#   # Direct single-process run:
#   ./tools/nsys_profile.sh ./build/examples/3dboxpec/3dboxpec examples/3dboxpec/3dboxpec.par
#   ./tools/nsys_profile.sh -o my_trace ./build/examples/3dboxpec/3dboxpec examples/3dboxpec/3dboxpec.par
#
#   # MPI run via -np flag (auto-invokes mpirun):
#   ./tools/nsys_profile.sh -np 2 -o mpi_trace ./build/examples/3dboxpec/3dboxpec examples/3dboxpec/3dboxpec.par
#
#   # MPI run directly inside mpirun:
#   mpirun -np 2 ./tools/nsys_profile.sh -o mpi_trace ./build/examples/3dboxpec/3dboxpec examples/3dboxpec/3dboxpec.par
#

# 1. Check if user passed -np or -n without already being inside mpirun
MPI_NP=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -np|-n)
            MPI_NP="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

# Locate mpirun if needed
MPIRUN_BIN=""
for candidate in \
    "$(which mpirun 2>/dev/null)" \
    "/local/spack/linux-centos8-zen2/gcc-12.2.0/openmpi-4.1.6-kvterypc5syr66ltu3zlkqnd4l6acycx/bin/mpirun" \
    "/usr/lib64/openmpi/bin/mpirun" \
    "/usr/lib64/mpich/bin/mpirun" \
    "/usr/local/bin/mpirun"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        MPIRUN_BIN="$candidate"
        MPI_DIR="$(dirname "$candidate")"
        export PATH="$MPI_DIR:$PATH"
        break
    fi
done

# If -np was supplied and we are NOT already inside an MPI rank, re-launch through mpirun
if [ -n "$MPI_NP" ] && [ -z "$OMPI_COMM_WORLD_RANK" ] && [ -z "$PMI_RANK" ] && [ -z "$SLURM_PROCID" ]; then
    if [ -z "$MPIRUN_BIN" ]; then
        echo "Error: Could not locate a working 'mpirun' executable."
        exit 1
    fi
    echo "=========================================================="
    echo " Launching $MPI_NP MPI ranks under mpirun for profiling..."
    echo " mpirun binary: $MPIRUN_BIN"
    echo "=========================================================="
    exec "$MPIRUN_BIN" --oversubscribe -np "$MPI_NP" "$0" "$@"
fi

# 2. Locate nsys binary
NSYS_BIN=""
for candidate in \
    "/usr/local/cuda-11.1/nsight-systems-2020.3.4/bin/nsys" \
    "/usr/local/cuda-11.1/nsight-systems-2020.3.4/target-linux-x64/nsys" \
    "/usr/local/cuda-11.7/bin/nsys" \
    "/usr/local/cuda/bin/nsys" \
    "/usr/local/bin/nsys" \
    "$(which nsys 2>/dev/null)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        if "$candidate" --version >/dev/null 2>&1; then
            NSYS_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$NSYS_BIN" ]; then
    echo "Error: Could not locate a working 'nsys' executable on this system."
    exit 1
fi

OUTPUT_NAME="nekwave_trace"

# 3. Parse optional -o / --output <name> flag
if [ "$1" == "-o" ] || [ "$1" == "--output" ]; then
    OUTPUT_NAME="$2"
    shift 2
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 [-np <ranks>] [-o <output_name>] <command> [args...]"
    echo "Example:"
    echo "  $0 -np 2 -o trace_3dboxpec ./build/examples/3dboxpec/3dboxpec examples/3dboxpec/3dboxpec.par"
    exit 1
fi

# 4. MPI rank detection to prevent output file collision across parallel ranks
RANK=""
if [ -n "$OMPI_COMM_WORLD_RANK" ]; then
    RANK="$OMPI_COMM_WORLD_RANK"
elif [ -n "$PMI_RANK" ]; then
    RANK="$PMI_RANK"
elif [ -n "$PMIX_RANK" ]; then
    RANK="$PMIX_RANK"
elif [ -n "$SLURM_PROCID" ]; then
    RANK="$SLURM_PROCID"
elif [ -n "$MV2_COMM_WORLD_RANK" ]; then
    RANK="$MV2_COMM_WORLD_RANK"
fi

if [ -n "$RANK" ]; then
    OUTPUT_NAME="${OUTPUT_NAME}_rank${RANK}"
fi

echo "=========================================================="
if [ -n "$RANK" ]; then
    echo " [MPI Rank $RANK] Running NVIDIA Nsight Systems Trace Profile"
else
    echo " Running NVIDIA Nsight Systems Trace Profile"
fi
echo " nsys binary: $NSYS_BIN"
echo " Output:      ${OUTPUT_NAME}.(qdrep/nsys-rep)"
echo " Trace:       cuda,nvtx,mpi,osrt"
echo " Target:      $*"
echo "=========================================================="

STATS_FLAG="--stats=true"
if [ -n "$RANK" ]; then
    STATS_FLAG="--stats=false"
fi

"$NSYS_BIN" profile \
    --trace=cuda,nvtx,mpi,osrt \
    --mpi-impl=openmpi \
    $STATS_FLAG \
    --force-overwrite=true \
    -o "$OUTPUT_NAME" \
    "$@"
