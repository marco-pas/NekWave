cmake_minimum_required(VERSION 3.16)

# =====================================================
# NekWave: C++11 Discontinuous Galerkin Maxwell Solver
# =====================================================
project(NekWave
    VERSION 0.1.0
    DESCRIPTION "High-fidelity time-domain Maxwell solver using DG-SEM"
    LANGUAGES CXX
)

# ------------------------------------------------------------------------------
# C++ Standard and Compiler Configuration
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type: Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

# Compiler warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-pedantic>
    )
endif()

# Ensure output directory exists in project root
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/output")

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(ENABLE_PROFILING "Build CPU profiler executable using gperftools" OFF)
option(BUILD_TESTING "Build unit tests and register with CTest" ON)
option(ENABLE_CUDA "Build CUDA-accelerated matrix multiplication kernel" OFF)

# ------------------------------------------------------------------------------
# CUDA Support (Optional Side-Compilation or Integration)
# ------------------------------------------------------------------------------
if(ENABLE_CUDA)
    # Check if CMAKE_CUDA_COMPILER is already set; if not, search common paths
    if(NOT CMAKE_CUDA_COMPILER)
        find_program(NVCC_EXECUTABLE NAMES nvcc
            PATHS /usr/local/cuda-11.7/bin /usr/local/cuda/bin
        )
        if(NVCC_EXECUTABLE)
            set(CMAKE_CUDA_COMPILER "${NVCC_EXECUTABLE}")
        endif()
    endif()

    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()

    add_library(nekwave_cuda STATIC
        src/cuda/mxm_cuda.cu
        src/cuda/gpu_solver.cu
    )
    target_include_directories(nekwave_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    )
    set_target_properties(nekwave_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# ------------------------------------------------------------------------------
# Core Solver Library (nekwave_core)
# ------------------------------------------------------------------------------
add_library(nekwave_core STATIC
    src/mesh.cpp
    src/physics.cpp
    src/timestepperrk45.cpp
    src/probe.cpp
    src/case.cpp
)

target_include_directories(nekwave_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

if(ENABLE_CUDA)
    target_link_libraries(nekwave_core PUBLIC nekwave_cuda)
    target_compile_definitions(nekwave_core PUBLIC NEKWAVE_ENABLE_CUDA)
endif()

# ------------------------------------------------------------------------------
# Main Application Executable (nekwave)
# ------------------------------------------------------------------------------
add_executable(nekwave src/main.cpp)
target_link_libraries(nekwave PRIVATE nekwave_core)
set_target_properties(nekwave PROPERTIES
    OUTPUT_NAME "nekwave"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ------------------------------------------------------------------------------
# CTest Integration and Test Registration
# ------------------------------------------------------------------------------
include(CTest)

if(BUILD_TESTING)
    # Binary name per docs/conventions.md: <module>-test
    add_executable(nekwave-test tests/nekwave_test.cpp)
    target_link_libraries(nekwave-test PRIVATE nekwave_core)
    set_target_properties(nekwave-test PROPERTIES
        OUTPUT_NAME "nekwave-test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    # CTest test names per docs/conventions.md: CamelCase ending with Tests
    add_test(NAME MeshQuadratureTests COMMAND nekwave-test MeshQuadratureTest)
    add_test(NAME ProbeScalingTests COMMAND nekwave-test ProbeScalingTest)
    add_test(NAME MaxwellStabilityTests COMMAND nekwave-test MaxwellPhysicsStabilityTest)
    add_test(NAME NekWaveUnitTests COMMAND nekwave-test)

    set_tests_properties(
        MeshQuadratureTests
        ProbeScalingTests
        MaxwellStabilityTests
        NekWaveUnitTests
        PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    if(ENABLE_CUDA)
        add_executable(nekwave_cuda_test src/cuda/test_mxm_cuda.cu)
        target_link_libraries(nekwave_cuda_test PRIVATE nekwave_cuda)
        set_target_properties(nekwave_cuda_test PROPERTIES
            OUTPUT_NAME "nekwave_cuda_test"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        )
        add_test(NAME CudaMatrixMultiplicationTests COMMAND nekwave_cuda_test)
        set_tests_properties(CudaMatrixMultiplicationTests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endif()
endif()

# ------------------------------------------------------------------------------
# Profiling Executable (gperftools)
# ------------------------------------------------------------------------------
if(ENABLE_PROFILING)
    find_path(GPERFTOOLS_INCLUDE_DIR gperftools/profiler.h
        PATHS /opt/homebrew/include /usr/local/include
    )
    find_library(GPERFTOOLS_PROFILER_LIB profiler
        PATHS /opt/homebrew/lib /usr/local/lib
    )

    if(GPERFTOOLS_PROFILER_LIB)
        add_executable(nekwave_prof src/main.cpp)
        target_link_libraries(nekwave_prof PRIVATE nekwave_core ${GPERFTOOLS_PROFILER_LIB})
        target_compile_options(nekwave_prof PRIVATE -g -fno-omit-frame-pointer)
        if(GPERFTOOLS_INCLUDE_DIR)
            target_include_directories(nekwave_prof PRIVATE ${GPERFTOOLS_INCLUDE_DIR})
        endif()
        set_target_properties(nekwave_prof PROPERTIES
            OUTPUT_NAME "nekwave_prof"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        )

        add_custom_target(profile
            COMMAND ${CMAKE_COMMAND} -E env CPUPROFILE=nekwave.prof $<TARGET_FILE:nekwave_prof> "${CMAKE_CURRENT_SOURCE_DIR}/examples/cavity_gaussian/cavity_gaussian.par"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            DEPENDS nekwave_prof
            COMMENT "Executing simulation with CPU profiling (output: nekwave.prof)..."
        )
    else()
        message(WARNING "gperftools profiler library not found. Profile target not configured.")
    endif()
endif()

# ------------------------------------------------------------------------------
# Custom Utility Targets: run and plot
# ------------------------------------------------------------------------------
add_custom_target(run
    COMMAND $<TARGET_FILE:nekwave> "${CMAKE_CURRENT_SOURCE_DIR}/examples/cavity_gaussian/cavity_gaussian.par"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS nekwave
    COMMENT "Running NekWave simulation with cavity_gaussian.par..."
)

# Detect Python interpreter for postprocessing
find_package(Python3 COMPONENTS Interpreter QUIET)
set(VENV_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python")
if(EXISTS "${VENV_PYTHON}")
    set(PYTHON_RUNNER "${VENV_PYTHON}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    set(PYTHON_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
elseif(Python3_EXECUTABLE)
    set(PYTHON_RUNNER "${Python3_EXECUTABLE}")
else()
    set(PYTHON_RUNNER "python3")
endif()

add_custom_target(plot
    COMMAND "${PYTHON_RUNNER}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/postprocess.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Executing Python postprocessing and generating modal analysis figures..."
)

# ------------------------------------------------------------------------------
# Examples Directory (Modular Case Suite)
# ------------------------------------------------------------------------------
add_subdirectory(examples)


