# ==============================================================================
# NekWave Automated Regression & Verification Tests
# ==============================================================================

# Core unit test harness
add_executable(nekwave-test nekwave_test.cpp)
target_link_libraries(nekwave-test PRIVATE nekwave_core)
set_target_properties(nekwave-test PROPERTIES
    OUTPUT_NAME "nekwave-test"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# 3D Box PEC Verification Test
add_executable(3dboxpec_test 3dboxpec_test/3dboxpec_test.cpp)
target_link_libraries(3dboxpec_test PRIVATE nekwave_core)
set_target_properties(3dboxpec_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ctests/3dboxpec_test"
)

# 3D Box Periodic Verification Test
add_executable(3dboxper_test 3dboxper_test/3dboxper_test.cpp)
target_link_libraries(3dboxper_test PRIVATE nekwave_core)
set_target_properties(3dboxper_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ctests/3dboxper_test"
)

# Data Saving Verification Test
add_executable(data_saving_test data_saving_test/data_saving_test.cpp)
target_link_libraries(data_saving_test PRIVATE nekwave_core)
set_target_properties(data_saving_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ctests/data_saving_test"
)

# Register CTest tests
if(BUILD_TESTING)
    # Unit tests
    add_test(NAME MeshQuadratureTests COMMAND nekwave-test MeshQuadratureTest)
    add_test(NAME ProbeScalingTests COMMAND nekwave-test ProbeScalingTest)
    add_test(NAME MaxwellStabilityTests COMMAND nekwave-test MaxwellStabilityTest)
    add_test(NAME CaseInputLoadingTests COMMAND nekwave-test CaseInputLoadingTest)
    add_test(NAME NekWaveUnitTests COMMAND nekwave-test)

    set_tests_properties(
        MeshQuadratureTests
        ProbeScalingTests
        MaxwellStabilityTests
        CaseInputLoadingTests
        NekWaveUnitTests
        PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )

    # Benchmark verification tests
    if(EXISTS "${CMAKE_SOURCE_DIR}/contrib/NekCEM/tests/3dboxpec/3dboxpec.rea")
        add_test(NAME BoxPecTests
            COMMAND 3dboxpec_test "${CMAKE_CURRENT_SOURCE_DIR}/3dboxpec_test/3dboxpec_test.par"
        )
        set_tests_properties(BoxPecTests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        )
    endif()

    if(EXISTS "${CMAKE_SOURCE_DIR}/contrib/NekCEM/tests/3dboxper/3dboxper.rea")
        add_test(NAME BoxPerTests
            COMMAND 3dboxper_test "${CMAKE_CURRENT_SOURCE_DIR}/3dboxper_test/3dboxper_test.par"
        )
        set_tests_properties(BoxPerTests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        )
    endif()

    # Data saving test
    add_test(NAME DataSavingTests
        COMMAND data_saving_test "${CMAKE_CURRENT_SOURCE_DIR}/data_saving_test/data_saving_test.par"
    )
    set_tests_properties(DataSavingTests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()
