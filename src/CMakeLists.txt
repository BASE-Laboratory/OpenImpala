# ==============================================================================
# OpenImpala - src/ CMakeLists.txt
# Builds the IO and Props library targets and the Diffusion application.
# ==============================================================================

# ==============================================================================
# IO Library
# ==============================================================================
# Library sources (exclude test drivers that start with 't')
add_library(openimpala_io OBJECT
    io/CathodeWrite.cpp
    io/DatReader.cpp
    io/HDF5Reader.cpp
    io/RawReader.cpp
    io/TiffReader.cpp
)

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
# Props Library (C++ and Fortran)
# ==============================================================================
add_library(openimpala_props OBJECT
    # C++ sources
    props/EffectiveDiffusivityHypre.cpp
    props/TortuosityDirect.cpp
    props/TortuosityHypre.cpp
    props/VolumeFraction.cpp
    # Fortran sources
    props/EffDiffFillMtx.F90
    props/TortuosityHypreFill.F90
    props/Tortuosity_filcc.F90
    props/Tortuosity_poisson_3d.F90
)

target_include_directories(openimpala_props PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/props
    ${CMAKE_CURRENT_SOURCE_DIR}/io
)

# Fortran module output directory
set_target_properties(openimpala_props PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran_modules
)

target_include_directories(openimpala_props PUBLIC
    ${CMAKE_BINARY_DIR}/fortran_modules
)

target_link_libraries(openimpala_props PUBLIC
    AMReX::amrex
    HYPRE::HYPRE
    MPI::MPI_CXX
    MPI::MPI_Fortran
    OpenMP::OpenMP_CXX
    OpenMP::OpenMP_Fortran
)

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
)

target_include_directories(Diffusion PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/io
    ${CMAKE_CURRENT_SOURCE_DIR}/props
    ${HDF5_INCLUDE_DIRS}
)

# Install the main executable
install(TARGETS Diffusion RUNTIME DESTINATION bin)
