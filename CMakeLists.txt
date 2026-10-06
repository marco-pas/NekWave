cmake_minimum_required(VERSION 3.16)

# =====================================================
# NekWave: C++11 Discontinuous Galerkin Maxwell Solver
# =====================================================
project(NekWave
    VERSION 0.1.0
    DESCRIPTION "High-fidelity time-domain Maxwell solver using DG-SEM"
    LANGUAGES C CXX
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

# Ensure output directory exists in the CMake build directory
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/output")

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(ENABLE_PROFILING "Build CPU profiler executable using gperftools" OFF)
option(BUILD_TESTING "Build unit tests and register with CTest" ON)
option(ENABLE_HIP "Build NekWave with AMD ROCm / HIP GPU backend instead of NVIDIA CUDA" OFF)

# ------------------------------------------------------------------------------
# GPU Backend: AMD ROCm / HIP or NVIDIA CUDA
# ------------------------------------------------------------------------------
if(ENABLE_HIP)
    message(STATUS "[NekWave] Configuring GPU backend: AMD ROCm / HIP")
    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH "$ENV{ROCM_PATH}")
        elseif(EXISTS "/opt/rocm")
            set(ROCM_PATH "/opt/rocm")
        else()
            set(ROCM_PATH "/opt/rocm")
        endif()
    endif()

    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        # Default to gfx90a (AMD Instinct MI210 / MI250 / MI250X, e.g. PDC Dardel / LUMI)
        set(CMAKE_HIP_ARCHITECTURES gfx90a CACHE STRING "HIP architectures")
    endif()
    if(NOT DEFINED AMDGPU_TARGETS)
        set(AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "AMD GPU targets to compile for")
    endif()
    if(NOT DEFINED GPU_TARGETS)
        set(GPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "GPU targets to compile for")
    endif()
    set(ENV{AMDGPU_TARGETS} "${CMAKE_HIP_ARCHITECTURES}")

    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip ${ROCM_PATH}/llvm)
    find_package(hip QUIET)

    if(hip_FOUND)
        message(STATUS "[NekWave] Found HIP: ${HIP_VERSION} (include: ${HIP_INCLUDE_DIRS})")
        set(DEVICE_INCLUDE_DIRS ${HIP_INCLUDE_DIRS} "${ROCM_PATH}/include")
    else()
        find_program(HIPCC_EXECUTABLE NAMES hipcc PATHS ${ROCM_PATH}/bin /opt/rocm/bin)
        set(DEVICE_INCLUDE_DIRS "${ROCM_PATH}/include")
    endif()

    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        enable_language(HIP)
        add_library(nekwave_device STATIC
            src/device/dg_kernels.cu
        )
        set_source_files_properties(src/device/dg_kernels.cu PROPERTIES LANGUAGE HIP)
        target_include_directories(nekwave_device PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${DEVICE_INCLUDE_DIRS}>
        )
        target_compile_definitions(nekwave_device PUBLIC USE_HIP=1 NEKWAVE_ENABLE_HIP=1)
        if(TARGET hip::device)
            target_link_libraries(nekwave_device PUBLIC hip::device)
        elseif(TARGET hip::host)
            target_link_libraries(nekwave_device PUBLIC hip::host)
        endif()
    else()
        add_library(nekwave_device STATIC
            src/device/dg_kernels.cu
        )
        set_target_properties(nekwave_device PROPERTIES
            LINKER_LANGUAGE CXX
            POSITION_INDEPENDENT_CODE ON
        )
        target_include_directories(nekwave_device PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${DEVICE_INCLUDE_DIRS}>
        )
        target_compile_definitions(nekwave_device PUBLIC USE_HIP=1 NEKWAVE_ENABLE_HIP=1)
    endif()

    set(CUDA_INCLUDE_DIR "")
    set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${ROCM_PATH}/lib;${ROCM_PATH}/lib/llvm/lib;${ROCM_PATH}/llvm/lib")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${ROCM_PATH}/lib;${ROCM_PATH}/lib/llvm/lib;${ROCM_PATH}/llvm/lib")
else()
    # ------------------------------------------------------------------------------
    # CUDA Support (Mandatory GPU Backend when not using HIP)
    # ------------------------------------------------------------------------------
    if(NOT CMAKE_CUDA_COMPILER OR NOT EXISTS "${CMAKE_CUDA_COMPILER}")
        find_program(NVCC_EXECUTABLE NAMES nvcc
            PATHS /usr/local/cuda-11.7/bin /usr/local/cuda/bin
            NO_DEFAULT_PATH
        )
        if(NOT NVCC_EXECUTABLE)
            find_program(NVCC_EXECUTABLE NAMES nvcc)
        endif()
        if(NVCC_EXECUTABLE)
            set(CMAKE_CUDA_COMPILER "${NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
        endif()
    endif()

    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "NekWave requires an NVIDIA GPU (CUDA) or AMD GPU (ROCm/HIP with -DENABLE_HIP=ON).")
    endif()

    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
    get_filename_component(CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT "${CUDA_BIN_DIR}" DIRECTORY)
    set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT}/include")

    add_library(nekwave_device STATIC
        src/device/dg_kernels.cu
    )
    target_include_directories(nekwave_device PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIR}>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    set_target_properties(nekwave_device PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# ------------------------------------------------------------------------------
# Auto-detect HPC / Spack Installations (e.g. PDC KTH Clusters)
# ------------------------------------------------------------------------------
if(NOT DEFINED CMAKE_PREFIX_PATH)
    set(SPACK_HDF5_DIR "/local/spack/linux-centos8-zen2/gcc-12.2.0/hdf5-1.14.3-2z7sqsie3axwefcpkcroig6xekuofspt")
    set(SPACK_MPI_DIR "/local/spack/linux-centos8-zen2/gcc-12.2.0/openmpi-4.1.6-kvterypc5syr66ltu3zlkqnd4l6acycx")
    if(EXISTS "${SPACK_HDF5_DIR}" AND EXISTS "${SPACK_MPI_DIR}")
        set(CMAKE_PREFIX_PATH ${SPACK_HDF5_DIR} ${SPACK_MPI_DIR})
    endif()
endif()

# ------------------------------------------------------------------------------
# MPI Support (Optional Multi-GPU / Multi-Node Parallelism)
# ------------------------------------------------------------------------------
option(ENABLE_MPI "Build NekWave with MPI multi-GPU support" ON)
if(ENABLE_MPI)
    find_package(MPI)
    if(MPI_FOUND)
        message(STATUS "[NekWave] Found MPI (C++): ${MPI_CXX_COMPILER}")
        set(NEKWAVE_MPI_LIBS MPI::MPI_CXX)
    else()
        message(STATUS "[NekWave] MPI not found; building in single-GPU / serial mode.")
        set(NEKWAVE_MPI_LIBS "")
    endif()
else()
    set(NEKWAVE_MPI_LIBS "")
endif()

# ------------------------------------------------------------------------------
# HDF5 Support (Native High-Performance Archive Output)
# ------------------------------------------------------------------------------
if(NOT DEFINED ENV{HDF5_DIR})
    file(GLOB CRAY_HDF5_DIRS
        "/opt/cray/pe/hdf5-parallel/*/*/*"
        "/opt/cray/pe/hdf5/*/*/*"
    )
    foreach(cand ${CRAY_HDF5_DIRS})
        if(EXISTS "${cand}/include/hdf5.h")
            set(ENV{HDF5_DIR} "${cand}")
            break()
        endif()
    endforeach()
endif()

if(DEFINED ENV{HDF5_DIR})
    set(HDF5_ROOT "$ENV{HDF5_DIR}")
    unset(HDF5_DIR CACHE)
    unset(HDF5_DIR)
endif()
set(HDF5_PREFER_PARALLEL TRUE)
set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE TRUE)
find_package(HDF5 COMPONENTS C QUIET)

# Fallback detection for Cray PE HDF5 or environment
if(NOT HDF5_FOUND AND DEFINED ENV{HDF5_DIR} AND EXISTS "$ENV{HDF5_DIR}/include/hdf5.h")
    message(STATUS "[NekWave] Using HDF5 from HDF5_DIR environment: $ENV{HDF5_DIR}")
    set(HDF5_FOUND TRUE)
    set(HDF5_INCLUDE_DIRS "$ENV{HDF5_DIR}/include")
    find_library(HDF5_C_LIB NAMES hdf5_parallel hdf5 PATHS "$ENV{HDF5_DIR}/lib" NO_DEFAULT_PATH)
    if(HDF5_C_LIB)
        set(NEKWAVE_HDF5_LIBS "${HDF5_C_LIB}")
    endif()
elseif(HDF5_FOUND)
    if(DEFINED ENV{HDF5_DIR} AND NOT HDF5_INCLUDE_DIRS)
        set(HDF5_INCLUDE_DIRS "$ENV{HDF5_DIR}/include")
    endif()
    message(STATUS "[NekWave] Found HDF5: ${HDF5_LIBRARIES} (include: ${HDF5_INCLUDE_DIRS})")
    set(NEKWAVE_HDF5_LIBS ${HDF5_LIBRARIES})
else()
    message(WARNING "[NekWave] Native HDF5 C library not found; binary pipeline will be used. On LUMI, please run: 'module load cray-hdf5-parallel'")
    set(NEKWAVE_HDF5_LIBS "")
endif()

if(HDF5_FOUND AND NOT NEKWAVE_HDF5_LIBS)
    find_library(HDF5_C_LIB NAMES hdf5_parallel hdf5
        PATHS
            "$ENV{HDF5_DIR}/lib"
            "$ENV{HDF5_ROOT}/lib"
            "$ENV{CRAY_HDF5_PARALLEL_DIR}/lib"
            /opt/cray/pe/hdf5-parallel/1.12.2.11/crayclang/17.0/lib
            /opt/cray/pe/hdf5-parallel/1.14.3.3/crayclang/17.0/lib
        NO_DEFAULT_PATH
    )
    if(NOT HDF5_C_LIB)
        find_library(HDF5_C_LIB NAMES hdf5_parallel hdf5)
    endif()
    if(HDF5_C_LIB)
        set(NEKWAVE_HDF5_LIBS "${HDF5_C_LIB}")
        message(STATUS "[NekWave] Explicitly linked HDF5 library: ${HDF5_C_LIB}")
    endif()
endif()

if(NEKWAVE_HDF5_LIBS)
    foreach(_lib ${NEKWAVE_HDF5_LIBS})
        if(EXISTS "${_lib}")
            get_filename_component(_lib_dir "${_lib}" DIRECTORY)
            set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${_lib_dir}")
            set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${_lib_dir}")
        endif()
    endforeach()
endif()

# ------------------------------------------------------------------------------
# Core Solver Library (nekwave_core)
# ------------------------------------------------------------------------------
set(NEKWAVE_CORE_SRCS
    src/comm.cpp
    src/boundary_conditions.cpp
    src/mesh.cpp
    src/dg_solver.cpp
    src/probe.cpp
    src/case.cpp
    src/hdf5_writer.cpp
)

add_library(nekwave_core STATIC
    ${NEKWAVE_CORE_SRCS}
)

target_include_directories(nekwave_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${DEVICE_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
)

if(HDF5_FOUND)
    target_include_directories(nekwave_core PUBLIC ${HDF5_INCLUDE_DIRS})
    target_compile_definitions(nekwave_core PUBLIC NEKWAVE_HAVE_HDF5=1)
    if(TARGET hdf5-shared)
        get_target_property(HDF5_LIB_LOC hdf5-shared LOCATION)
        if(HDF5_LIB_LOC)
            get_filename_component(HDF5_LIB_DIR "${HDF5_LIB_LOC}" DIRECTORY)
            set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${HDF5_LIB_DIR}")
            set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${HDF5_LIB_DIR}")
        endif()
    endif()
endif()

if(MPI_FOUND AND ENABLE_MPI)
    target_include_directories(nekwave_core PUBLIC ${MPI_CXX_INCLUDE_PATH})
    target_compile_definitions(nekwave_core PUBLIC NEKWAVE_ENABLE_MPI=1)
    target_link_libraries(nekwave_core PUBLIC ${NEKWAVE_MPI_LIBS})
    get_filename_component(MPI_LIB_DIR "${MPI_CXX_LIBRARIES}" DIRECTORY)
    if(MPI_LIB_DIR)
        link_directories(${MPI_LIB_DIR})
        set(CMAKE_INSTALL_RPATH "${MPI_LIB_DIR}")
        set(CMAKE_BUILD_RPATH "${MPI_LIB_DIR}")
    endif()

    if(ENABLE_HIP)
        # On Cray EX systems with ROCm, GPU-aware MPI requires the GTL (GPU Transport Layer) library: libmpi_gtl_hsa
        find_library(CRAY_MPI_GTL_HSA_LIB
            NAMES mpi_gtl_hsa
            PATHS
                /opt/cray/pe/lib64
                /opt/cray/pe/mpich/8.1.32/gtl/lib
                "$ENV{CRAY_MPICH_DIR}/../gtl/lib"
                "$ENV{MPICH_DIR}/../gtl/lib"
        )
        if(CRAY_MPI_GTL_HSA_LIB)
            message(STATUS "[NekWave] Linking Cray MPICH GPU Transport Layer (GTL): ${CRAY_MPI_GTL_HSA_LIB}")
            target_link_libraries(nekwave_core PUBLIC ${CRAY_MPI_GTL_HSA_LIB})
            get_filename_component(GTL_LIB_DIR "${CRAY_MPI_GTL_HSA_LIB}" DIRECTORY)
            set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${GTL_LIB_DIR}")
            set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${GTL_LIB_DIR}")
        endif()
    endif()
endif()

if(ENABLE_HIP)
    target_compile_definitions(nekwave_core PUBLIC
        NEKWAVE_BUILD_DIR="${CMAKE_BINARY_DIR}"
        NEKWAVE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        USE_HIP=1
        NEKWAVE_ENABLE_HIP=1
    )
else()
    target_compile_definitions(nekwave_core PUBLIC
        NEKWAVE_BUILD_DIR="${CMAKE_BINARY_DIR}"
        NEKWAVE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        NEKWAVE_ENABLE_CUDA
    )
endif()

target_link_libraries(nekwave_core PUBLIC nekwave_device ${NEKWAVE_HDF5_LIBS} ${CMAKE_DL_LIBS})

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
    add_subdirectory(tests)
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
            COMMAND ${CMAKE_COMMAND} -E env CPUPROFILE=nekwave.prof $<TARGET_FILE:nekwave_prof> "${CMAKE_CURRENT_SOURCE_DIR}/examples/3dboxpec/3dboxpec.json"
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
    COMMAND $<TARGET_FILE:nekwave> "${CMAKE_CURRENT_SOURCE_DIR}/examples/3dboxpec/3dboxpec.json"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS nekwave
    COMMENT "Running NekWave simulation with 3dboxpec.json..."
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


