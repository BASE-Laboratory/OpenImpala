/** @file SolverConfig.H
 *  @brief Shared solver configuration helpers: string-to-enum conversion for solver types
 *         and flow directions.
 *
 *  Eliminates the duplicated stringToSolverType() and stringToDirection() helper functions
 *  that were previously copy-pasted across Diffusion.cpp, tMultiPhaseTransport.cpp,
 *  tEffectiveDiffusivity.cpp, tTortuosity.cpp, and tTortuosityDirect.cpp.
 */
#ifndef SOLVER_CONFIG_H
#define SOLVER_CONFIG_H

#include "Tortuosity.H"

#include <AMReX_Print.H>

#include <algorithm>
#include <string>

namespace OpenImpala {

/** @brief Return true when solver_str selects the matrix-free AMReX MLMG solver ("mlmg" or "auto").
 */
inline bool isMLMGSolverString(const std::string& solver_str) {
    std::string s = solver_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return (s == "mlmg" || s == "auto");
}

/** @brief Convert a case-insensitive solver name to SolverType.
 *  @param solver_str  Solver name (e.g. "FlexGMRES", "pcg", "BiCGSTAB").
 *  @return Corresponding SolverType enum value.
 *  @note Calls amrex::Abort if solver_str is unrecognised.
 */
inline SolverType parseSolverType(const std::string& solver_str) {
    std::string s = solver_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "jacobi")
        return SolverType::Jacobi;
    if (s == "gmres")
        return SolverType::GMRES;
    if (s == "flexgmres")
        return SolverType::FlexGMRES;
    if (s == "pcg")
        return SolverType::PCG;
    if (s == "bicgstab")
        return SolverType::BiCGSTAB;
    if (s == "smg")
        return SolverType::SMG;
    if (s == "pfmg")
        return SolverType::PFMG;
    amrex::Abort("Invalid solver string: '" + solver_str + "'.");
    return SolverType::FlexGMRES;
}

/** @brief Backward-compatible alias for parseSolverType().
 *  @deprecated Use parseSolverType() instead.
 */
inline SolverType tortuositySolverType(const std::string& solver_str) {
    return parseSolverType(solver_str);
}

/** @brief Backward-compatible alias for parseSolverType().
 *  @deprecated Use parseSolverType() instead.
 */
inline SolverType effDiffSolverType(const std::string& solver_str) {
    return parseSolverType(solver_str);
}

/** @brief Convert a case-insensitive direction string to Direction enum.
 *  @param dir_str  Direction name ("X", "Y", or "Z", case-insensitive).
 *  @return Corresponding Direction enum value.
 *  @note Calls amrex::Abort if dir_str is unrecognised.
 */
inline Direction parseDirection(const std::string& dir_str) {
    std::string s = dir_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "x")
        return Direction::X;
    if (s == "y")
        return Direction::Y;
    if (s == "z")
        return Direction::Z;
    amrex::Abort("Invalid direction string: " + dir_str + ". Use X, Y, or Z.");
    return Direction::X;
}

/** @brief Convert a case-insensitive precision string to MLMGPrecision enum. */
inline MLMGPrecision parseMLMGPrecision(const std::string& prec_str) {
    std::string s = prec_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "float64" || s == "double" || s == "fp64")
        return MLMGPrecision::Float64;
    if (s == "float32" || s == "single" || s == "fp32")
        return MLMGPrecision::Float32;
    if (s == "mixed")
        return MLMGPrecision::Mixed;
    amrex::Abort("Invalid MLMG precision string: '" + prec_str +
                 "'. Use 'float64', 'float32', or 'mixed'.");
    return MLMGPrecision::Float64;
}

/** @brief Convert an MLMGPrecision enum value to its canonical lowercase name. */
inline const char* mlmgPrecisionString(MLMGPrecision prec) {
    switch (prec) {
    case MLMGPrecision::Float64:
        return "float64";
    case MLMGPrecision::Float32:
        return "float32";
    case MLMGPrecision::Mixed:
        return "mixed";
    default:
        return "float64";
    }
}

} // namespace OpenImpala

#endif // SOLVER_CONFIG_H
