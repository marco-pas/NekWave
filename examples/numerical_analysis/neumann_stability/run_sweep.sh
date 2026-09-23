#!/bin/bash
# ==============================================================================
# NekWave Neumann Stability Sweep
# Runs all combinations of N = {4, 6, 8, 10} and C0 = {0.0, 0.5, 1.0}
# Overlays the spatial eigenvalues on the LSRK45 stability footprint.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
RESULTS_DIR="$SCRIPT_DIR/results"

# Default CFL number to scale eigenvalues (z = lambda * dt)
SAFETY=0.4

if [ ! -f "$BUILD_DIR/ctests/numerical_analysis/analytical_export/analytical_export" ]; then
    echo "Error: C++ executable not found!"
    echo "Please compile the project first in the build/ directory."
    exit 1
fi

# Create results folder
mkdir -p "$RESULTS_DIR"

# Move to build directory to run the export cleanly
cd "$BUILD_DIR" || exit 1

for N in 2 3 4 5 6 7 8 9 10; do
    for C0 in 0.0 0.5 1.0; do
        JSON_FILE="matrices_N${N}_C${C0}.json"
        
        echo "=========================================================="
        echo " Generating Neumann Stability Plot: N = $N, C0 = $C0, Safety = $SAFETY"
        echo "=========================================================="
        
        # 1. Export matrices from C++
        ./ctests/numerical_analysis/analytical_export/analytical_export "$N" "$C0" "$JSON_FILE"
        
        # 2. Compute eigenvalues and plot overlay on RK45 footprint
        python3 ../examples/numerical_analysis/neumann_stability/plot_neumann.py "$JSON_FILE" "$SAFETY"
        
        # 3. Move resulting plot to results directory
        if [ -f "results/neumann_stability.png" ]; then
            mv results/neumann_stability.png "$RESULTS_DIR/neumann_N${N}_C${C0}.png"
            echo "-> Saved to examples/numerical_analysis/neumann_stability/results/neumann_N${N}_C${C0}.png"
        else
            echo "-> Error: Plot was not generated!"
        fi
        
        # Cleanup temporary JSON
        rm -f "$JSON_FILE"
    done
done

# Cleanup empty results dir created in build by python script
rmdir results 2>/dev/null || true

echo "=========================================================="
echo "Sweep complete! All 12 Neumann Stability plots are saved in:"
echo "$RESULTS_DIR"
echo "=========================================================="
