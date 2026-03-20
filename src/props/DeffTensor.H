/** @file DeffTensor.H
 *  @brief Computes the effective diffusivity tensor from homogenization corrector fields.
 *
 *  Given the three corrector fields chi_x, chi_y, chi_z (solutions of the cell problem
 *  nabla_xi . (D nabla_xi chi_k) = -nabla_xi . (D e_k)), assembles the full 3x3
 *  effective diffusivity tensor via volume averaging:
 *
 *      D_eff[i][j] = (1/V) * sum_active{ delta_ij + d(chi_j)/d(x_i) }
 */
#ifndef DEFF_TENSOR_H
#define DEFF_TENSOR_H

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_REAL.H>

namespace OpenImpala {

/** @brief Assembles the effective diffusivity tensor from corrector fields.
 *
 *  Computes D_eff[i][j] = (1/|Omega|) * integral_Omega { delta_ij + d(chi_j)/d(x_i) } dV
 *  using central differences on the chi fields and masking by the active phase.
 *
 *  @param[out] Deff_tensor   3x3 output tensor (row-major: Deff_tensor[row][col]).
 *  @param[in]  mf_chi_x      Corrector field for x-direction (1 component, >= 1 ghost cell).
 *  @param[in]  mf_chi_y      Corrector field for y-direction (1 component, >= 1 ghost cell).
 *  @param[in]  mf_chi_z      Corrector field for z-direction (1 component, >= 1 ghost cell).
 *  @param[in]  active_mask   Integer mask: 1 = active cell, 0 = inactive (0 ghost cells).
 *  @param[in]  geom          AMReX Geometry defining cell sizes for gradient computation.
 *  @param[in]  verbose       Verbosity level (0 = silent).
 */
void calculateDeffTensor(amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM],
                         const amrex::MultiFab& mf_chi_x, const amrex::MultiFab& mf_chi_y,
                         const amrex::MultiFab& mf_chi_z, const amrex::iMultiFab& active_mask,
                         const amrex::Geometry& geom, int verbose = 0);

} // namespace OpenImpala

#endif // DEFF_TENSOR_H
