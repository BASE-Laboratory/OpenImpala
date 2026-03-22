/** @file HypreCheck.H
 *  @brief Shared HYPRE error-checking macro.
 */
#ifndef OPENIMPALA_HYPRECHECK_H_
#define OPENIMPALA_HYPRECHECK_H_

#include <AMReX_BLassert.H>
#include <HYPRE_utilities.h>
#include <string>

#define HYPRE_CHECK(ierr)                                                                          \
    do {                                                                                           \
        if ((ierr) != 0) {                                                                         \
            char hypre_error_msg[256];                                                             \
            HYPRE_DescribeError(ierr, hypre_error_msg);                                            \
            amrex::Abort("HYPRE Error: " + std::string(hypre_error_msg) +                          \
                         " - Error Code: " + std::to_string(ierr) + " File: " + __FILE__ +         \
                         " Line: " + std::to_string(__LINE__));                                    \
        }                                                                                          \
    } while (0)

#endif // OPENIMPALA_HYPRECHECK_H_
