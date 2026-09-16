# ==============================================================================
# NekWave Examples (Neko-style Modular Case Suite)
# ==============================================================================

# ------------------------------------------------------------------------------
# Example 1: Cavity Gaussian Pulse Benchmark
# ------------------------------------------------------------------------------
add_executable(cavity_gaussian cavity_gaussian/cavity_gaussian.cpp)
target_link_libraries(cavity_gaussian PRIVATE nekwave_core)
set_target_properties(cavity_gaussian PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/cavity_gaussian"
)

# ------------------------------------------------------------------------------
# Example 2: Analytical Cavity TM_110 Eigenmode Benchmark
# ------------------------------------------------------------------------------
add_executable(cavity_eigenmode cavity_eigenmode/cavity_eigenmode.cpp)
target_link_libraries(cavity_eigenmode PRIVATE nekwave_core)
set_target_properties(cavity_eigenmode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/cavity_eigenmode"
)

# ------------------------------------------------------------------------------
# CTest Registrations for Examples
# ------------------------------------------------------------------------------
if(BUILD_TESTING)
    add_test(NAME CavityGaussianRun
        COMMAND cavity_gaussian "${CMAKE_CURRENT_SOURCE_DIR}/cavity_gaussian/cavity_gaussian.par"
    )
    set_tests_properties(CavityGaussianRun PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cavity_gaussian"
    )

    add_test(NAME CavityEigenmodeRun
        COMMAND cavity_eigenmode "${CMAKE_CURRENT_SOURCE_DIR}/cavity_eigenmode/cavity_eigenmode.par"
    )
    set_tests_properties(CavityEigenmodeRun PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cavity_eigenmode"
    )
endif()

