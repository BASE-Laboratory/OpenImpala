# FindHYPRE.cmake
# ----------------
# Find the HYPRE library (https://github.com/hypre-space/hypre).
#
# This module is needed when HYPRE was built with autoconf (./configure)
# rather than CMake, so no HYPREConfig.cmake is installed.
#
# The following variables influence the search:
#   HYPRE_ROOT / HYPRE_HOME / ENV{HYPRE_HOME}
#
# This module defines:
#   HYPRE_FOUND        - True if HYPRE was found
#   HYPRE_INCLUDE_DIRS - HYPRE include directories
#   HYPRE_LIBRARIES    - HYPRE libraries to link against
#
# And the imported target:
#   HYPRE::HYPRE

# Look for the header
find_path(HYPRE_INCLUDE_DIR
    NAMES HYPRE.h
    HINTS
        ${HYPRE_ROOT}
        ${HYPRE_HOME}
        $ENV{HYPRE_ROOT}
        $ENV{HYPRE_HOME}
        $ENV{HYPRE_INSTALL_PREFIX}
    PATH_SUFFIXES include
)

# Look for the library
find_library(HYPRE_LIBRARY
    NAMES HYPRE
    HINTS
        ${HYPRE_ROOT}
        ${HYPRE_HOME}
        $ENV{HYPRE_ROOT}
        $ENV{HYPRE_HOME}
        $ENV{HYPRE_INSTALL_PREFIX}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HYPRE
    REQUIRED_VARS HYPRE_LIBRARY HYPRE_INCLUDE_DIR
)

if(HYPRE_FOUND)
    set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR})
    set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})

    if(NOT TARGET HYPRE::HYPRE)
        add_library(HYPRE::HYPRE UNKNOWN IMPORTED)
        set_target_properties(HYPRE::HYPRE PROPERTIES
            IMPORTED_LOCATION "${HYPRE_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(HYPRE_INCLUDE_DIR HYPRE_LIBRARY)
