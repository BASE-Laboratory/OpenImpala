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
openimpala_add_test(tPercolationCheck       "${CMAKE_CURRENT_SOURCE_DIR}/tPercolationCheck.cpp")

# ==============================================================================
# I/O Tests — Zero Coverage Targets (Issue #109)
# ==============================================================================
openimpala_add_test(tDatReader      "${CMAKE_CURRENT_SOURCE_DIR}/tDatReader.cpp")
openimpala_add_test(tCathodeWrite   "${CMAKE_CURRENT_SOURCE_DIR}/tCathodeWrite.cpp")

# ==============================================================================
# Solver Tests — Zero Coverage Target (Issue #109)
# ==============================================================================
openimpala_add_test(tTortuosityDirect  "${CMAKE_CURRENT_SOURCE_DIR}/tTortuosityDirect.cpp")

# ==============================================================================
# Synthetic Data Tests — Improve Partial Coverage (Issue #109)
# ==============================================================================
openimpala_add_test(tSyntheticVolumeFraction  "${CMAKE_CURRENT_SOURCE_DIR}/tSyntheticVolumeFraction.cpp")
openimpala_add_test(tSyntheticPercolation     "${CMAKE_CURRENT_SOURCE_DIR}/tSyntheticPercolation.cpp")
openimpala_add_test(tSyntheticEffectiveDiffusivity "${CMAKE_CURRENT_SOURCE_DIR}/tSyntheticEffectiveDiffusivity.cpp")
openimpala_add_test(tSyntheticMicrostructure "${CMAKE_CURRENT_SOURCE_DIR}/tSyntheticMicrostructure.cpp")

# ==============================================================================
# Multi-Phase Transport Tests
# ==============================================================================

# Synthetic multi-phase test with analytical verification (uniform domain, tau=1.0)
openimpala_add_test(tMultiPhaseTransport    "${CMAKE_CURRENT_SOURCE_DIR}/tMultiPhaseTransport.cpp")

# Two-phase synthetic variant (alternating layers, both D=1.0, tau=1.0)
add_test(
    NAME tMultiPhaseTransport_twophase
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tMultiPhaseTransport_twophase.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tMultiPhaseTransport_twophase PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# Multi-phase variant of tortuosity test (reuses tTortuosity executable with multi-phase inputs)
add_test(
    NAME tTortuosity_multiphase
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tTortuosity>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tTortuosity_multiphase.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tTortuosity_multiphase PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# Multi-phase variant of effective diffusivity test (reuses tEffectiveDiffusivity executable)
add_test(
    NAME tEffectiveDiffusivity_multiphase
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tEffectiveDiffusivity>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tEffectiveDiffusivity_multiphase.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tEffectiveDiffusivity_multiphase PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# ==============================================================================
# Directional Symmetry Tests (Y and Z directions)
# ==============================================================================

# Y direction — uniform domain should give same tau as X
add_test(
    NAME tMultiPhaseTransport_dirY
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tMultiPhaseTransport_dirY.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tMultiPhaseTransport_dirY PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# Z direction — uniform domain should give same tau as X
add_test(
    NAME tMultiPhaseTransport_dirZ
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tMultiPhaseTransport_dirZ.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tMultiPhaseTransport_dirZ PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# ==============================================================================
# Physics Type Tests (Electrical Conductivity)
# ==============================================================================

# Electrical conductivity variant of tortuosity test (reuses tTortuosity executable)
add_test(
    NAME tTortuosity_conductivity
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tTortuosity>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tTortuosity_conductivity.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tTortuosity_conductivity PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# Electrical conductivity variant of synthetic multi-phase test
add_test(
    NAME tMultiPhaseTransport_conductivity
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tMultiPhaseTransport_conductivity.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tMultiPhaseTransport_conductivity PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# ==============================================================================
# Physics Type Tests (Thermal Conductivity)
# ==============================================================================

# Thermal conductivity variant with bulk_property = 401 W/(m·K) (copper)
add_test(
    NAME tMultiPhaseTransport_thermal
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tMultiPhaseTransport_thermal.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tMultiPhaseTransport_thermal PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# ==============================================================================
# Boundary Condition Tests (Issue #16)
#
# Verify that the BC abstraction layer produces correct results:
#   dirichlet_external_bc — explicit DirichletExternal matches default
#   periodic_sides        — Periodic lateral BCs on uniform block
# ==============================================================================

# Explicit DirichletExternal BC: same result as default
add_test(
    NAME tMultiPhaseTransport_dirichlet_external_bc
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tMultiPhaseTransport_dirichlet_external_bc.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tMultiPhaseTransport_dirichlet_external_bc PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# Periodic lateral BCs on uniform block
add_test(
    NAME tMultiPhaseTransport_periodic_sides
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tMultiPhaseTransport_periodic_sides.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tMultiPhaseTransport_periodic_sides PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# ==============================================================================
# Regression Benchmarks (Issue #80)
#
# Lightweight synthetic geometries with known analytical solutions.
# These guard against solver regressions: if any of these fail, the code
# is producing incorrect physics.
#
#   uniform  — homogeneous block, tau = (N-1)/N
#   series   — layers perpendicular to flow (Reuss/harmonic bound)
#   parallel — layers parallel to flow (Voigt/arithmetic bound)
# ==============================================================================

# Uniform dense block: single-phase, tau = 0.96875
add_test(
    NAME tRegressionBenchmark_uniform
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tRegressionBenchmark_uniform.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tRegressionBenchmark_uniform PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# Series resistance: D_eff = harmonic mean, tau = 1.453125
add_test(
    NAME tRegressionBenchmark_series
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tRegressionBenchmark_series.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tRegressionBenchmark_series PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# Parallel resistance: D_eff = arithmetic mean, tau = 1.291667
add_test(
    NAME tRegressionBenchmark_parallel
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tRegressionBenchmark_parallel.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tRegressionBenchmark_parallel PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# ==============================================================================
# PCG Solver Regression Benchmarks (Issue #109)
#
# Validates the PCG solver path (new in this branch) against the same
# analytical solutions used for FlexGMRES benchmarks.
# ==============================================================================

# PCG on uniform block: tau = (N-1)/N = 0.96875
add_test(
    NAME tRegressionBenchmark_uniform_PCG
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tRegressionBenchmark_uniform_PCG.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tRegressionBenchmark_uniform_PCG PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# PCG on series layers: tau = 1.453125
add_test(
    NAME tRegressionBenchmark_series_PCG
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tMultiPhaseTransport>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tRegressionBenchmark_series_PCG.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tRegressionBenchmark_series_PCG PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# ==============================================================================
# Diffusion.cpp Integration Tests (Issue #109)
#
# Exercise the main application entry point (src/props/Diffusion.cpp),
# which is the largest file at 0% coverage.  Uses the Diffusion executable
# built by src/CMakeLists.txt.
# ==============================================================================

# Dry-run mode: exercises file I/O, volume fraction, percolation checks
add_test(
    NAME tDiffusion_dryrun
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:Diffusion>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tDiffusion_dryrun.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tDiffusion_dryrun PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# Full TIFF solve: exercises TIFF reader branch + homogenization pipeline
add_test(
    NAME tDiffusion_tiff
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:Diffusion>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tDiffusion_tiff.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tDiffusion_tiff PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 600
)

# HDF5 solve: exercises HDF5 reader branch
add_test(
    NAME tDiffusion_hdf5
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:Diffusion>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tDiffusion_hdf5.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tDiffusion_hdf5 PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 600
)

# ==============================================================================
# FloodFill Tests (Issue #170 — shared GPU-compatible flood fill utility)
# ==============================================================================
openimpala_add_test(tFloodFill  "${CMAKE_CURRENT_SOURCE_DIR}/tFloodFill.cpp")

# ==============================================================================
# TortuosityMLMG Tests (Issue #171 — matrix-free MLMG solver)
#
# Validates the AMReX-native MLMG solver against the same analytical
# solutions used for the HYPRE-based solver.
# ==============================================================================
openimpala_add_test(tTortuosityMLMG  "${CMAKE_CURRENT_SOURCE_DIR}/tTortuosityMLMG.cpp")

# MLMG uniform block: tau = (N-1)/N = 0.96875
add_test(
    NAME tTortuosityMLMG_uniform
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tTortuosityMLMG>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tTortuosityMLMG_uniform.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tTortuosityMLMG_uniform PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# MLMG Y-direction symmetry: same tau as X
add_test(
    NAME tTortuosityMLMG_dirY
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:tTortuosityMLMG>
            ${CMAKE_SOURCE_DIR}/tests/inputs/tTortuosityMLMG_dirY.inputs
            amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(tTortuosityMLMG_dirY PROPERTIES
    ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    TIMEOUT 300
)

# ==============================================================================
# Unit Tests (Catch2 — fast, no MPI/AMReX required)
# ==============================================================================
add_subdirectory(unit)
