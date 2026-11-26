#!/usr/bin/env bash
set -euxo pipefail

# 1. Build static PIC AMReX 25.03 into a staging prefix if AMReXConfig.cmake is not already in $PREFIX
AMREX_STAGE="${SRC_DIR}/_amrex_install"
if [ ! -f "${PREFIX}/lib/cmake/AMReX/AMReXConfig.cmake" ]; then
    cmake -S "${SRC_DIR}/amrex" -B "${SRC_DIR}/_amrex_build" -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${AMREX_STAGE}" \
        -DAMReX_SPACEDIM=3 \
        -DAMReX_MPI=ON \
        -DAMReX_OMP=ON \
        -DAMReX_FORTRAN=OFF \
        -DAMReX_LINEAR_SOLVERS=ON \
        -DAMReX_EB=OFF \
        -DAMReX_TINY_PROFILE=ON \
        -DAMReX_PIC=ON \
        -DBUILD_SHARED_LIBS=OFF
    cmake --build "${SRC_DIR}/_amrex_build" -j"${CPU_COUNT:-2}"
    cmake --install "${SRC_DIR}/_amrex_build"
    export CMAKE_PREFIX_PATH="${AMREX_STAGE}:${PREFIX}:${CMAKE_PREFIX_PATH:-}"
else
    export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"
fi

# 2. Build and install the standalone C++/MPI Diffusion driver
cmake -S "${SRC_DIR}/openimpala" -B "${SRC_DIR}/_oi_cli_build" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DBUILD_TESTING=OFF \
    -DOPENIMPALA_PYTHON=OFF
cmake --build "${SRC_DIR}/_oi_cli_build" -j"${CPU_COUNT:-2}"
cmake --install "${SRC_DIR}/_oi_cli_build"

# 3. Build and install the Python openimpala package (_core pybind11 extension)
export SETUPTOOLS_SCM_PRETEND_VERSION="${PKG_VERSION}"
"${PYTHON}" -m pip install "${SRC_DIR}/openimpala" \
    --no-deps --no-build-isolation -vv
