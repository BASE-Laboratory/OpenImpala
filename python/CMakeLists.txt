# ==============================================================================
# OpenImpala Python Bindings — CMakeLists.txt
# ==============================================================================
#
# Builds the _core pybind11 extension module, linking against the existing
# openimpala_io and openimpala_props OBJECT libraries.  AMReX types are
# managed natively — no pyamrex dependency.
#
# Usage (from the top-level build directory):
#   cmake .. -DOPENIMPALA_PYTHON=ON
#   make -j$(nproc)
#
# Or via pip / scikit-build-core:
#   pip install .
# ==============================================================================

cmake_minimum_required(VERSION 3.18)

# --- Find Python and pybind11 ---
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 2.11 CONFIG REQUIRED)

# --- Build the extension module ---
set(BINDING_SOURCES
    bindings/module.cpp
    bindings/enums.cpp
    bindings/io.cpp
    bindings/props.cpp
    bindings/solvers.cpp
    bindings/config.cpp
)
pybind11_add_module(_core ${BINDING_SOURCES})

# When CUDA is enabled, AMReX headers contain __host__/__device__ attributes
# that require compilation by nvcc.
if(GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(${BINDING_SOURCES} PROPERTIES LANGUAGE CUDA)
    # AMReX's static library contains relocatable device code (RDC).  The _core
    # shared library must run CUDA's device linker to resolve __cudaRegister*
    # symbols from AMReX and HYPRE device objects.
    set_target_properties(_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

target_link_libraries(_core PRIVATE
    openimpala_io
    openimpala_props
    AMReX::amrex
    HYPRE::HYPRE
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_C_LIBRARIES}
    TIFF::TIFF
    MPI::MPI_CXX
    nlohmann_json::nlohmann_json
)

target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings
    ${CMAKE_SOURCE_DIR}/src/io
    ${CMAKE_SOURCE_DIR}/src/props
    ${HDF5_INCLUDE_DIRS}
)

# Compile with C++17 to match the rest of the project
target_compile_features(_core PRIVATE cxx_std_17)

# --- Place the compiled extension directly into the Python package ---
# This ensures Python's relative imports (from . import _core) work 
# natively during local testing. We skip this during wheel builds 
# (when SKBUILD is defined) so scikit-build-core can track the artifact.
if(NOT SKBUILD)
    set_target_properties(_core PROPERTIES 
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openimpala"
    )
endif()

# --- Install into the Python package layout ---
install(TARGETS _core LIBRARY DESTINATION openimpala COMPONENT python)
install(FILES
    openimpala/__init__.py
    openimpala/session.py
    openimpala/exceptions.py
    openimpala/facade.py
    openimpala/cli.py
    DESTINATION openimpala
    COMPONENT python
)

# ==============================================================================
# Python Tests (pytest via CTest)
# ==============================================================================
# Register pytest as a CTest target.  The PYTHONPATH must include both the
# build directory (for _core.so) and the source directory (for the pure-Python
# package).  Tests are only registered when BUILD_TESTING is ON.
if(BUILD_TESTING)
    # Discover pytest executable
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pytest --version
        RESULT_VARIABLE PYTEST_FOUND
        OUTPUT_QUIET ERROR_QUIET
    )

    if(PYTEST_FOUND EQUAL 0)
        set(PYTHON_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        set(PYTHON_PKG_DIR  "${CMAKE_CURRENT_SOURCE_DIR}")

        # Helper: register one pytest file as a CTest test
        function(openimpala_add_pytest TEST_NAME TEST_FILE)
            add_test(
                NAME ${TEST_NAME}
                COMMAND ${Python_EXECUTABLE} -m pytest
                        ${TEST_FILE}
                        -v --tb=short
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            )
            set_tests_properties(${TEST_NAME} PROPERTIES
                ENVIRONMENT "PYTHONPATH=${PYTHON_PKG_DIR}:$ENV{PYTHONPATH}"
                TIMEOUT 120
                LABELS "python"
            )
        endfunction()
        
        # --- Pure-Python tests (no compiled module needed) ---
        openimpala_add_pytest(pytest_cli
            "${PYTHON_TEST_DIR}/test_cli.py")

        # --- Tests requiring _core module ---
        openimpala_add_pytest(pytest_enums
            "${PYTHON_TEST_DIR}/test_enums.py")
        openimpala_add_pytest(pytest_facade_helpers
            "${PYTHON_TEST_DIR}/test_facade_helpers.py")
        openimpala_add_pytest(pytest_config
            "${PYTHON_TEST_DIR}/test_config.py")
        openimpala_add_pytest(pytest_volume_fraction
            "${PYTHON_TEST_DIR}/test_volume_fraction.py")
        openimpala_add_pytest(pytest_percolation
            "${PYTHON_TEST_DIR}/test_percolation.py")
        openimpala_add_pytest(pytest_tortuosity
            "${PYTHON_TEST_DIR}/test_tortuosity.py")

        message(STATUS "Python tests registered with CTest (label: python)")
    else()
        message(STATUS "pytest not found — Python tests will not be registered with CTest")
    endif()
endif()
