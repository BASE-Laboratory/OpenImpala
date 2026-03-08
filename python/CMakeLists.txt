# ==============================================================================
# OpenImpala Python Bindings — CMakeLists.txt
# ==============================================================================
#
# Builds the _core pybind11 extension module, linking against the existing
# openimpala_io and openimpala_props OBJECT libraries and pyAMReX for AMReX
# type interop.
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
pybind11_add_module(_core
    bindings/module.cpp
    bindings/enums.cpp
    bindings/io.cpp
    bindings/props.cpp
    bindings/solvers.cpp
    bindings/config.cpp
)

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
    ${CMAKE_SOURCE_DIR}/src/io
    ${CMAKE_SOURCE_DIR}/src/props
    ${HDF5_INCLUDE_DIRS}
)

# Compile with C++17 to match the rest of the project
target_compile_features(_core PRIVATE cxx_std_17)

# --- Install into the Python package layout ---
install(TARGETS _core LIBRARY DESTINATION openimpala)
install(FILES
    openimpala/__init__.py
    openimpala/session.py
    openimpala/exceptions.py
    openimpala/facade.py
    openimpala/cli.py
    DESTINATION openimpala
)
