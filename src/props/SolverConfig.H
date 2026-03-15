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

#include "EffectiveDiffusivityHypre.H"
#include "Tortuosity.H"
#include "TortuosityHypre.H"

#include <AMReX_Print.H>

#include <algorithm>
#include <string>

namespace OpenImpala {

/** @brief Convert a case-insensitive solver name to TortuosityHypre::SolverType.
 *  @param solver_str  Solver name (e.g. "FlexGMRES", "pcg", "BiCGSTAB").
 *  @return Corresponding SolverType enum value.
 *  @note Calls amrex::Abort if solver_str is unrecognised.
 */
inline TortuosityHypre::SolverType tortuositySolverType(const std::string& solver_str) {
    std::string s = solver_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "jacobi")
        return TortuosityHypre::SolverType::Jacobi;
    if (s == "gmres")
        return TortuosityHypre::SolverType::GMRES;
    if (s == "flexgmres")
        return TortuosityHypre::SolverType::FlexGMRES;
    if (s == "pcg")
        return TortuosityHypre::SolverType::PCG;
    if (s == "bicgstab")
        return TortuosityHypre::SolverType::BiCGSTAB;
    if (s == "smg")
        return TortuosityHypre::SolverType::SMG;
    if (s == "pfmg")
        return TortuosityHypre::SolverType::PFMG;
    amrex::Abort("Invalid solver string: '" + solver_str + "'.");
    return TortuosityHypre::SolverType::FlexGMRES;
}

/** @brief Convert a case-insensitive solver name to EffectiveDiffusivityHypre::SolverType.
 *  @param solver_str  Solver name (e.g. "FlexGMRES", "pcg", "BiCGSTAB").
 *  @return Corresponding SolverType enum value.
 *  @note Calls amrex::Abort if solver_str is unrecognised.
 */
inline EffectiveDiffusivityHypre::SolverType effDiffSolverType(const std::string& solver_str) {
    std::string s = solver_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "jacobi")
        return EffectiveDiffusivityHypre::SolverType::Jacobi;
    if (s == "gmres")
        return EffectiveDiffusivityHypre::SolverType::GMRES;
    if (s == "flexgmres")
        return EffectiveDiffusivityHypre::SolverType::FlexGMRES;
    if (s == "pcg")
        return EffectiveDiffusivityHypre::SolverType::PCG;
    if (s == "bicgstab")
        return EffectiveDiffusivityHypre::SolverType::BiCGSTAB;
    if (s == "smg")
        return EffectiveDiffusivityHypre::SolverType::SMG;
    if (s == "pfmg")
        return EffectiveDiffusivityHypre::SolverType::PFMG;
    amrex::Abort("Invalid solver string: '" + solver_str + "'.");
    return EffectiveDiffusivityHypre::SolverType::FlexGMRES;
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

} // namespace OpenImpala

#endif // SOLVER_CONFIG_H
