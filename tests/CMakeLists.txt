# ==============================================================================
# OpenImpala - tests/ CMakeLists.txt
# Builds test executables and registers them with CTest.
#
# Test source files live in this directory (prefixed with 't').
# Each test is an independent executable that links against the project libraries.
# Tests are run via: mpirun -np 1 <test_exe> <inputs_file>
# ==============================================================================

# Common link libraries for all tests
set(OPENIMPALA_TEST_LIBS
    openimpala_io
    openimpala_props
    AMReX::amrex
    HYPRE::HYPRE
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_C_LIBRARIES}
    TIFF::TIFF
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

set(OPENIMPALA_TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/src/io
    ${CMAKE_SOURCE_DIR}/src/props
    ${HDF5_INCLUDE_DIRS}
)

# Helper: define a test executable and register it with CTest.
# If an inputs file exists in tests/inputs/<name>.inputs, it is passed as an argument.
function(openimpala_add_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE ${OPENIMPALA_TEST_LIBS})
    target_include_directories(${TEST_NAME} PRIVATE ${OPENIMPALA_TEST_INCLUDES})

    # Check for a matching inputs file
    set(INPUTS_FILE "${CMAKE_SOURCE_DIR}/tests/inputs/${TEST_NAME}.inputs")
    if(EXISTS "${INPUTS_FILE}")
        add_test(
            NAME ${TEST_NAME}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:${TEST_NAME}>
                    ${INPUTS_FILE}
                    amrex.verbose=0
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    else()
        add_test(
            NAME ${TEST_NAME}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:${TEST_NAME}>
                    amrex.verbose=0
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endif()

    # Allow running as root inside containers
    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
        TIMEOUT 300
    )
endfunction()

# ==============================================================================
# IO Tests
# ==============================================================================
openimpala_add_test(tTiffReader    "${CMAKE_CURRENT_SOURCE_DIR}/tTiffReader.cpp")
openimpala_add_test(tRawReader     "${CMAKE_CURRENT_SOURCE_DIR}/tRawReader.cpp")
openimpala_add_test(tHDF5Reader    "${CMAKE_CURRENT_SOURCE_DIR}/tHDF5Reader.cpp")

# ==============================================================================
# Props Tests
# ==============================================================================
openimpala_add_test(tTortuosity             "${CMAKE_CURRENT_SOURCE_DIR}/tTortuosity.cpp")
openimpala_add_test(tVolumeFraction         "${CMAKE_CURRENT_SOURCE_DIR}/tVolumeFraction.cpp")
openimpala_add_test(tEffectiveDiffusivity   "${CMAKE_CURRENT_SOURCE_DIR}/tEffectiveDiffusivity.cpp")
