#!/bin/bash
set -e

# Create new directory structure
mkdir -p examples/numerical_analysis
mkdir -p examples/numerical_analysis/analytical_export
mkdir -p examples/numerical_analysis/neumann_stability
mkdir -p examples/numerical_analysis/full_dispersion
mkdir -p examples/numerical_analysis/numerical_dispersion

# 1. Move the C++ exporter
mv examples/analytical_spatial_dispersion/analytical_spatial_export.cpp examples/numerical_analysis/analytical_export/analytical_export.cpp

# 2. Move Numerical Dispersion
mv examples/numerical_dispersion/* examples/numerical_analysis/numerical_dispersion/
rmdir examples/numerical_dispersion

# 3. Move Full Dispersion Roadmap
mv examples/analytical_full_dispersion/* examples/numerical_analysis/full_dispersion/
rmdir examples/analytical_full_dispersion

# 4. Move RK45 Plotter to Neumann Stability
mv examples/analytical_time_dispersion/plot_stability.py examples/numerical_analysis/neumann_stability/plot_neumann.py
rm -rf examples/analytical_time_dispersion

# 5. Move Python spatial solver logic to tools/ ? Or leave it in tools/
# The tools/analytical_spatial_dispersion.py script is the spatial solver. We can leave it in tools/ or move it to examples/numerical_analysis/full_dispersion/. The user didn't explicitly mention moving tools/, but it's cleaner to keep python analysis scripts in tools/.

# Remove old spatial dispersion folder
rm -rf examples/analytical_spatial_dispersion

# Fix CMakeLists.txt
cat examples/CMakeLists.txt | \
sed 's|analytical_spatial_dispersion/analytical_spatial_export.cpp|numerical_analysis/analytical_export/analytical_export.cpp|g' | \
sed 's|analytical_spatial_export|analytical_export|g' | \
sed 's|analytical_spatial_dispersion|numerical_analysis/analytical_export|g' | \
sed 's|numerical_dispersion/numerical_dispersion.cpp|numerical_analysis/numerical_dispersion/numerical_dispersion.cpp|g' \
> tmp.cmake
mv tmp.cmake examples/CMakeLists.txt

# Create Master README
cat << 'README' > examples/numerical_analysis/README.md
# NekWave Numerical Analysis Suite

This directory contains the tools and studies for analyzing the numerical properties (stability and dispersion) of the NekWave DGTD solver.

## 1. Neumann Stability (`neumann_stability/`)
Analyzes the stability of the numerical scheme by computing the exact spatial eigenvalues of the reference element DGTD operators ($M$, $S$, $F$) and overlaying them on the stability region of the Low-Storage RK45 time-integration scheme. Used to determine maximum stable CFL numbers for different numerical fluxes (Central, Upwind, etc.).

## 2. Full Analytical Dispersion (`full_dispersion/`)  *(WIP)*
Computes the exact theoretical spatio-temporal dispersion relation (phase velocity vs. resolution). Combines the spatial Bloch-Floquet amplification matrix with the RK45 stability polynomial to predict numerical dissipation and dispersion errors analytically.

## 3. Numerical Dispersion (`numerical_dispersion/`) *(WIP)*
Validates the theoretical dispersion by running actual 3D CUDA simulations of a single periodic element. Extracts the numerical frequency via FFT of the time-domain probe history and compares it against the exact physical wave speed.

## Common Tools
- `analytical_export/`: A C++ utility that directly hooks into the NekWave core to export the exact reference element mass and stiffness matrices to JSON for the Python analytical scripts.
README

