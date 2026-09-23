#!/usr/bin/env bash
#
# NekWave Nsight Systems Profiling Helper
#
# Usage:
#   ../tools/nsys_profile.sh ./examples/cavity_gaussian/cavity_gaussian cavity_gaussian.par
#   ../tools/nsys_profile.sh -o my_trace ./examples/cavity_gaussian/cavity_gaussian cavity_gaussian.par
#

# Locate nsys binary
NSYS_BIN=""
for candidate in \
    "/usr/local/cuda-11.1/nsight-systems-2020.3.4/bin/nsys" \
    "/usr/local/cuda-11.1/nsight-systems-2020.3.4/target-linux-x64/nsys" \
    "$(which nsys 2>/dev/null)"; do
    if [ -x "$candidate" ]; then
        # Check if candidate actually runs
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

# Parse optional -o <name> flag
if [ "$1" == "-o" ] || [ "$1" == "--output" ]; then
    OUTPUT_NAME="$2"
    shift 2
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 [-o output_name] <command> [args...]"
    echo "Example:"
    echo "  $0 ./examples/cavity_gaussian/cavity_gaussian ../examples/cavity_gaussian/cavity_gaussian.par"
    exit 1
fi

echo "=========================================================="
echo " Running NVIDIA Nsight Systems Trace Profile"
echo " nsys binary: $NSYS_BIN"
echo " Output:      $OUTPUT_NAME.qdrep"
echo " Target:      $@"
echo "=========================================================="

"$NSYS_BIN" profile \
    --trace=cuda,osrt \
    --stats=true \
    --force-overwrite=true \
    -o "$OUTPUT_NAME" \
    "$@"

