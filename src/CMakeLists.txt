# ==============================================================================
# OpenImpala - src/ CMakeLists.txt
# Builds the IO and Props library targets and the Diffusion application.
# ==============================================================================

# ==============================================================================
# IO Library
# ==============================================================================
# Library sources (exclude test drivers that start with 't')
set(IO_SOURCES
    io/CathodeWrite.cpp
    io/DatReader.cpp
    io/HDF5Reader.cpp
    io/ImageLoader.cpp
    io/RawReader.cpp
    io/TiffReader.cpp
)
add_library(openimpala_io OBJECT ${IO_SOURCES})

target_include_directories(openimpala_io PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/io
)

target_link_libraries(openimpala_io PUBLIC
    AMReX::amrex
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_C_LIBRARIES}
    TIFF::TIFF
)

target_include_directories(openimpala_io PUBLIC
    ${HDF5_INCLUDE_DIRS}
)

# ==============================================================================
# Props Library (C++ only — Fortran kernels migrated to native C++)
# ==============================================================================
set(PROPS_SOURCES
    props/ConnectedComponents.cpp
    props/DeffTensor.cpp
    props/EffectiveDiffusivityHypre.cpp
    props/FloodFill.cpp
    props/HypreStructSolver.cpp
    props/ParticleSizeDistribution.cpp
    props/PercolationCheck.cpp
    props/REVStudy.cpp
    props/SpecificSurfaceArea.cpp
    props/ThroughThicknessProfile.cpp
    props/TortuosityDirect.cpp
    props/TortuosityHypre.cpp
    props/TortuosityMLMG.cpp
    props/TortuositySolverBase.cpp
    props/VolumeFraction.cpp
)
add_library(openimpala_props OBJECT ${PROPS_SOURCES})

target_include_directories(openimpala_props PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/props
    ${CMAKE_CURRENT_SOURCE_DIR}/io
)

target_link_libraries(openimpala_props PUBLIC
    AMReX::amrex
    HYPRE::HYPRE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

# ==============================================================================
# Position-Independent Code (required for pybind11 shared library linking)
# ==============================================================================
if(OPENIMPALA_PYTHON)
    set_target_properties(openimpala_io openimpala_props PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# ==============================================================================
# CUDA Separable Compilation (required for device linking with AMReX/HYPRE)
# ==============================================================================
if(GPU_BACKEND STREQUAL "CUDA")
    set_target_properties(openimpala_io openimpala_props PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ==============================================================================
# Diffusion Application
# ==============================================================================
add_executable(Diffusion
    props/Diffusion.cpp
)

target_link_libraries(Diffusion PRIVATE
    openimpala_io
    openimpala_props
    AMReX::amrex
    HYPRE::HYPRE
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_C_LIBRARIES}
    TIFF::TIFF
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    nlohmann_json::nlohmann_json
)

target_include_directories(Diffusion PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/io
    ${CMAKE_CURRENT_SOURCE_DIR}/props
    ${HDF5_INCLUDE_DIRS}
)

# ==============================================================================
# CUDA: compile all C++ sources as CUDA when GPU backend is enabled
# ==============================================================================
# AMReX headers use __host__/__device__ attributes (AMREX_GPU_HOST_DEVICE)
# when built with CUDA. Any translation unit that includes AMReX headers must
# therefore be compiled by nvcc. We achieve this by setting the LANGUAGE
# property to CUDA on all .cpp source files.
if(GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(
        ${IO_SOURCES} ${PROPS_SOURCES} props/Diffusion.cpp
        PROPERTIES LANGUAGE CUDA
    )
endif()

# Install the main executable
install(TARGETS Diffusion RUNTIME DESTINATION bin)
