# ==============================================================================
# NekWave Examples (Neko-style Modular Case Suite)
# ==============================================================================

# ------------------------------------------------------------------------------
# Example 1: NekCEM 3D Box PEC Benchmark Case
# ------------------------------------------------------------------------------
add_executable(3dboxpec 3dboxpec/3dboxpec.cpp)
target_link_libraries(3dboxpec PRIVATE nekwave_core)
set_target_properties(3dboxpec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/3dboxpec"
)

# ------------------------------------------------------------------------------
# Example 2: NekCEM 3D Box Periodic Benchmark Case
# ------------------------------------------------------------------------------
add_executable(3dboxper 3dboxper/3dboxper.cpp)
target_link_libraries(3dboxper PRIVATE nekwave_core)
set_target_properties(3dboxper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/3dboxper"
)

# ------------------------------------------------------------------------------
# Example 3: Numerical Dispersion Benchmark (k - omega)
# ------------------------------------------------------------------------------
add_executable(numerical_dispersion numerical_analysis/numerical_dispersion/numerical_dispersion.cpp)
target_link_libraries(numerical_dispersion PRIVATE nekwave_core)
set_target_properties(numerical_dispersion PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/numerical_dispersion"
)

# ------------------------------------------------------------------------------
# Example 4: Analytical Dispersion Exporter
# ------------------------------------------------------------------------------
add_executable(analytical_export numerical_analysis/analytical_export/analytical_export.cpp)
target_link_libraries(analytical_export PRIVATE nekwave_core)
set_target_properties(analytical_export PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/numerical_analysis/analytical_export"
)

# ------------------------------------------------------------------------------
# Example 5: High-Order Spatial Convergence Benchmark (h- and p-Refinement)
# ------------------------------------------------------------------------------
add_executable(convergence_test numerical_analysis/convergence_test/convergence_test.cpp)
target_link_libraries(convergence_test PRIVATE nekwave_core)
set_target_properties(convergence_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/numerical_analysis/convergence_test"
)

# ------------------------------------------------------------------------------
# Example 6: 3D PEC Sphere Scattering (RCS) with UPML Absorbing Boundaries
# ------------------------------------------------------------------------------
add_executable(sphereRCS sphereRCS/sphereRCS.cpp)
target_link_libraries(sphereRCS PRIVATE nekwave_core)
set_target_properties(sphereRCS PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/sphereRCS"
)
