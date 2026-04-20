# ==============================================================================
# OpenImpala - CMake Build System
# ==============================================================================
#
# Usage (inside the Singularity/Apptainer container):
#   mkdir build && cd build
#   cmake ..
#   make -j$(nproc)
#   ctest
#
# Usage (native build with custom dependency paths):
#   cmake .. -DCMAKE_PREFIX_PATH="/path/to/amrex;/path/to/hypre;/path/to/hdf5;/path/to/libtiff"
#
# ==============================================================================

cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# Derive project version from the latest Git tag (e.g. v4.0.1 → 4.0.1).
# Falls back to 0.0.0 when building outside a Git repository or when no
# tag is reachable (e.g. shallow clone without --tags).
# ---------------------------------------------------------------------------
set(OPENIMPALA_FALLBACK_VERSION "4.0.1")

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" describe --tags --match "v[0-9]*" --abbrev=0
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE _git_tag
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _git_result
    )
    if(_git_result EQUAL 0 AND _git_tag MATCHES "^v([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(_detected_version "${CMAKE_MATCH_1}")
    endif()
endif()

if(NOT _detected_version)
    set(_detected_version "${OPENIMPALA_FALLBACK_VERSION}")
    message(STATUS "Git tag not found — using fallback version ${_detected_version}")
else()
    message(STATUS "Version from Git tag: ${_detected_version}")
endif()

project(OpenImpala
    VERSION ${_detected_version}
    LANGUAGES C CXX Fortran
    DESCRIPTION "Image-based simulation of transport properties in porous media"
)

# ==============================================================================
# GPU Acceleration
# ==============================================================================
# Set GPU_BACKEND to CUDA, HIP, or NONE (default) to enable GPU-accelerated
# HYPRE solves. When enabled, AMReX and HYPRE must also be built with matching
# GPU support. The CUDA or HIP language is added to the project automatically.
set(GPU_BACKEND "NONE" CACHE STRING "GPU backend: NONE, CUDA, or HIP")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS NONE CUDA HIP)

if(GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(OPENIMPALA_USE_GPU OPENIMPALA_USE_CUDA)
    message(STATUS "GPU acceleration enabled: CUDA")
elseif(GPU_BACKEND STREQUAL "HIP")
    enable_language(HIP)
    add_compile_definitions(OPENIMPALA_USE_GPU OPENIMPALA_USE_HIP)
    message(STATUS "GPU acceleration enabled: HIP")
else()
    message(STATUS "GPU acceleration: disabled")
endif()

# ==============================================================================
# C++ Standard
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# Build type defaults
# ==============================================================================
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

# ==============================================================================
# Custom CMake modules
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ==============================================================================
# Dependencies
# ==============================================================================

# --- nlohmann/json (header-only, for structured JSON output) ---
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# --- MPI ---
find_package(MPI REQUIRED COMPONENTS CXX)

# --- OpenMP ---
find_package(OpenMP REQUIRED COMPONENTS CXX)

# --- AMReX ---
# AMReX ships AMReXConfig.cmake when installed via CMake.
# The container sets CMAKE_PREFIX_PATH to include the AMReX install prefix.
find_package(AMReX REQUIRED)

# --- HYPRE ---
# HYPRE is built with autoconf in the container, so we use our own FindHYPRE.cmake.
# If HYPRE was built with CMake and installed, find_package(HYPRE CONFIG) would work instead.
find_package(HYPRE REQUIRED)

# --- HDF5 ---
# We need the C++ bindings (hdf5_cpp).
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# --- Transitive dependencies for static TIFF ---
find_package(ZLIB REQUIRED)
find_package(JPEG REQUIRED)

# libtiff 4.6.0 statically links the math library and exports CMath::CMath.
# We must manually define this target so CMake can resolve TIFF::tiff.
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY AND NOT TARGET CMath::CMath)
    add_library(CMath::CMath UNKNOWN IMPORTED)
    set_target_properties(CMath::CMath PROPERTIES IMPORTED_LOCATION "${MATH_LIBRARY}")
endif()

# --- TIFF ---
find_package(TIFF REQUIRED)

# ==============================================================================
# Compiler flags
# ==============================================================================
# AMReX requires -DOMPI_SKIP_MPICXX to prevent inclusion of the deprecated MPI C++ bindings.
add_compile_definitions(OMPI_SKIP_MPICXX)

# ==============================================================================
# Code Coverage
# ==============================================================================
option(ENABLE_COVERAGE "Enable code coverage instrumentation (GCC/Clang --coverage)" OFF)
option(OPENIMPALA_PYTHON "Build Python bindings via pybind11" OFF)
if(ENABLE_COVERAGE)
    message(STATUS "Code coverage enabled — appending --coverage to compile and link flags")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

# ==============================================================================
# TinyProfiler
# ==============================================================================
# AMReX's TinyProfiler emits a function-level timing table at AMReX::Finalize().
# Useful for diagnosing C++ hotspots from the profiling notebook (§7).
#
# This option assumes AMReX was itself built with AMReX_TINY_PROFILE=ON — if not,
# BL_PROFILE regions compile away and no table is emitted. The wheel CI builds
# AMReX with this flag when OPENIMPALA_ENABLE_TINY_PROFILE=ON.
option(OPENIMPALA_ENABLE_TINY_PROFILE
       "Enable AMReX TinyProfiler instrumentation (BL_PROFILE regions)" OFF)
if(OPENIMPALA_ENABLE_TINY_PROFILE)
    add_compile_definitions(AMREX_TINY_PROFILE)
    message(STATUS "AMReX TinyProfiler: ENABLED (AMReX must also be built with "
                   "AMReX_TINY_PROFILE=ON for tables to appear)")
endif()

# ==============================================================================
# Library targets
# ==============================================================================
add_subdirectory(src)

# ==============================================================================
# Testing
# ==============================================================================
include(CTest)
if(BUILD_TESTING)
    # --- Catch2 (for lightweight unit tests) ---
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

    add_subdirectory(tests)
endif()

# ==============================================================================
# Python Bindings
# ==============================================================================
if(OPENIMPALA_PYTHON)
    add_subdirectory(python)
endif()
