/** @file DeffTensor.H
 *  @brief Computes the effective diffusivity tensor from homogenization corrector fields.
 *
 *  Given the three corrector fields chi_x, chi_y, chi_z (solutions of the periodic
 *  conservative finite-volume cell problem -div(D (e_k + grad chi_k)) = 0),
 *  assembles the symmetric positive semi-definite 3x3 effective diffusivity tensor
 *  by averaging harmonic face fluxes across the periodic unit cell:
 *
 *      D_eff[r][c] = (1/N_total) * sum_{x in Omega} D_{x+1/2}^{(r)} * (delta_rc + (chi_c(x+e_r) -
 * chi_c(x)) / h_r)
 */
#ifndef DEFF_TENSOR_H
#define DEFF_TENSOR_H

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_REAL.H>

namespace OpenImpala {

/** @brief Assembles the effective diffusivity tensor from periodic corrector fields.
 *
 *  Integrates conservative harmonic face fluxes across all +x, +y, +z faces in the
 *  periodic domain, preserving exact discrete Onsager symmetry and Voigt/Reuss bounds.
 *
 *  @param[out] Deff_tensor    3x3 output tensor (row-major: Deff_tensor[row][col]).
 *  @param[in]  mf_chi_x       Corrector field for x-direction (1 component, >= 1 ghost cell).
 *  @param[in]  mf_chi_y       Corrector field for y-direction (1 component, >= 1 ghost cell).
 *  @param[in]  mf_chi_z       Corrector field for z-direction (1 component, >= 1 ghost cell).
 *  @param[in]  active_mask    Integer mask: 1 = active cell, 0 = inactive (>= 0 ghost cells).
 *  @param[in]  geom           Periodic AMReX Geometry defining cell sizes and periodicity.
 *  @param[in]  verbose        Verbosity level (0 = silent).
 *  @param[in]  mf_diff_coeff  Optional cell-centred diffusivity MultiFab (defaults to 1.0 on active
 * cells).
 */
void calculateDeffTensor(amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM],
                         const amrex::MultiFab& mf_chi_x, const amrex::MultiFab& mf_chi_y,
                         const amrex::MultiFab& mf_chi_z, const amrex::iMultiFab& active_mask,
                         const amrex::Geometry& geom, int verbose = 0,
                         const amrex::MultiFab* mf_diff_coeff = nullptr);

} // namespace OpenImpala

#endif // DEFF_TENSOR_H
