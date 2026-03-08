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

project(OpenImpala
    VERSION 0.1.0
    LANGUAGES C CXX Fortran
    DESCRIPTION "Image-based simulation of transport properties in porous media"
)

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
find_package(MPI REQUIRED COMPONENTS CXX Fortran)

# --- OpenMP ---
find_package(OpenMP REQUIRED COMPONENTS CXX Fortran)

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
