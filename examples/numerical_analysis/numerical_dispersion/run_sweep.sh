#!/bin/bash
# ==============================================================================
# NekWave Two-Probe Numerical Dispersion CUDA Benchmark Sweep
# 10x10x10 Periodic Box Benchmark across N in {2..10} and C0 in {0.0, 0.5, 1.0}
# Measures phase velocity purely from time-domain probe histories.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
RESULTS_DIR="$SCRIPT_DIR/results"

PAR_FILE="$SCRIPT_DIR/numerical_dispersion.par"
EXE="$BUILD_DIR/ctests/numerical_dispersion/numerical_dispersion"

if [ ! -f "$PAR_FILE" ]; then
    echo "Error: Parameter file not found at $PAR_FILE"
    exit 1
fi

if [ ! -f "$EXE" ]; then
    echo "Error: Executable not found at $EXE. Compile first in build/"
    exit 1
fi

mkdir -p "$RESULTS_DIR"
cd "$BUILD_DIR" || exit 1

for N in 2 3 4 5 6 7 8 9 10; do
    MODES=5

    for C0 in 0.0 0.5 1.0; do
        echo "=========================================================="
        echo " Running Two-Probe Benchmark: N = $N, C0 = $C0, Modes = $MODES (k = 0.25 .. 3.75)"
        echo "=========================================================="
        
        # 1. Update order, flux parameter, and modes in .par file
        sed -i -E "s/^order = [0-9]+/order = $N/" "$PAR_FILE"
        sed -i -E "s/^C0 = [0-9.]+/C0 = $C0/" "$PAR_FILE"
        sed -i -E "s/^num_modes = [0-9]+/num_modes = $MODES/" "$PAR_FILE"
        
        # 2. Run C++ / CUDA simulation
        "$EXE" "$PAR_FILE"
        
        # 3. Analyze probe_history.csv using two-probe modal phase difference
        OUT_PLOT="$RESULTS_DIR/dispersion_N${N}_C${C0}.png"
        python3 "$SCRIPT_DIR/plot_two_probe_dispersion.py" output/probe_history.csv "$OUT_PLOT"
        
    done
done

echo "=========================================================="
echo "Sweep complete! All Two-Probe numerical plots are saved in:"
echo "$RESULTS_DIR"
echo "=========================================================="
