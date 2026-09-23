# ==============================================================================
# NekWave Examples (Neko-style Modular Case Suite)
# ==============================================================================

# ------------------------------------------------------------------------------
# Example 1: Cavity Gaussian Pulse Benchmark
# ------------------------------------------------------------------------------
add_executable(cavity_gaussian cavity_gaussian/cavity_gaussian.cpp)
target_link_libraries(cavity_gaussian PRIVATE nekwave_core)
set_target_properties(cavity_gaussian PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ctests/cavity_gaussian"
)

# ------------------------------------------------------------------------------
# Example 2: Analytical Cavity TM_110 Eigenmode Benchmark
# ------------------------------------------------------------------------------
add_executable(cavity_eigenmode cavity_eigenmode/cavity_eigenmode.cpp)
target_link_libraries(cavity_eigenmode PRIVATE nekwave_core)
set_target_properties(cavity_eigenmode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ctests/cavity_eigenmode"
)

# ------------------------------------------------------------------------------
# Example 3: Numerical Dispersion Benchmark (k - omega)
# ------------------------------------------------------------------------------
add_executable(numerical_dispersion numerical_analysis/numerical_dispersion/numerical_dispersion.cpp)
target_link_libraries(numerical_dispersion PRIVATE nekwave_core)
set_target_properties(numerical_dispersion PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ctests/numerical_dispersion"
)

# ------------------------------------------------------------------------------
# Example 4: Analytical Dispersion Exporter
# ------------------------------------------------------------------------------
add_executable(analytical_export numerical_analysis/analytical_export/analytical_export.cpp)
target_link_libraries(analytical_export PRIVATE nekwave_core)
set_target_properties(analytical_export PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ctests/numerical_analysis/analytical_export"
)

# ------------------------------------------------------------------------------
# CTest Registrations for Examples
# ------------------------------------------------------------------------------
if(BUILD_TESTING)
    # CTest test names per docs/conventions.md: CamelCase ending with Tests
    # Uses read-only test fixtures from tests/input with outputs disabled
    add_test(NAME CavityGaussianTests
        COMMAND cavity_gaussian "${CMAKE_SOURCE_DIR}/tests/input/cavity_gaussian.par"
    )
    set_tests_properties(CavityGaussianTests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    add_test(NAME CavityEigenmodeTests
        COMMAND cavity_eigenmode "${CMAKE_SOURCE_DIR}/tests/input/cavity_eigenmode.par"
    )
    set_tests_properties(CavityEigenmodeTests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    add_test(NAME NumericalDispersionTests
        COMMAND numerical_dispersion "${CMAKE_SOURCE_DIR}/tests/input/numerical_dispersion.par"
    )
    set_tests_properties(NumericalDispersionTests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()
