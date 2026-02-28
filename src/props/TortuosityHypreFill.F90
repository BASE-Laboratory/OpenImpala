module tortuosity_poisson_3d_module

  use amrex_fort_module, only : amrex_real
  use amrex_error_module, only : amrex_abort

  implicit none

  private ! Default module visibility to private

  public :: tortuosity_fillmtx ! Make only the subroutine public

  integer, parameter :: direction_x = 0
  integer, parameter :: direction_y = 1
  integer, parameter :: direction_z = 2

  integer, parameter :: comp_phase = 1
  integer, parameter :: comp_mask = 1

  ! Stencil indices (0-based, matching C++ HYPRE_StructStencilSetElement order)
  integer, parameter :: istn_c  = 0 ! Center
  integer, parameter :: istn_mx = 1 ! -X (West)
  integer, parameter :: istn_px = 2 ! +X (East)
  integer, parameter :: istn_my = 3 ! -Y (South)
  integer, parameter :: istn_py = 4 ! +Y (North)
  integer, parameter :: istn_mz = 5 ! -Z (Bottom)
  integer, parameter :: istn_pz = 6 ! +Z (Top)
  integer, parameter :: nstencil = 7

  ! Activity mask values (assuming C++ convention)
  integer, parameter :: cell_inactive = 0
  integer, parameter :: cell_active = 1

  ! Tolerance for floating point comparisons
  real(amrex_real), parameter :: small_real = 1.0e-15_amrex_real

contains

  ! ::: -----------------------------------------------------------
  ! ::: Fills HYPRE matrix coefficients for a variable-coefficient
  ! ::: diffusion equation: div(D(x) grad(phi)) = 0
  ! ::: Uses harmonic mean for face coefficients between cells.
  ! ::: Supports multi-phase computations with spatially varying D.
  ! ::: -----------------------------------------------------------
  subroutine tortuosity_fillmtx(a, rhs, xinit, nval, &
                                p, p_lo, p_hi, &
                                active_mask, mask_lo, mask_hi, &
                                diff_coeff, dc_lo, dc_hi, &
                                bxlo, bxhi, domlo, domhi, dxinv, vlo, vhi, phase, dir, &
                                debug_print_level) bind(c)

    ! Argument declarations
    integer,            intent(in)  :: nval
    real(amrex_real),   intent(out) :: a(0:nval*nstencil-1), rhs(nval), xinit(nval)
    integer,            intent(in)  :: p_lo(3), p_hi(3)
    integer,            intent(in)  :: p(p_lo(1):p_hi(1), p_lo(2):p_hi(2), p_lo(3):p_hi(3), *)
    integer,            intent(in)  :: mask_lo(3), mask_hi(3)
    integer,            intent(in)  :: active_mask(mask_lo(1):mask_hi(1), &
                                                     mask_lo(2):mask_hi(2), &
                                                     mask_lo(3):mask_hi(3))
    integer,            intent(in)  :: dc_lo(3), dc_hi(3)
    real(amrex_real),   intent(in)  :: diff_coeff(dc_lo(1):dc_hi(1), &
                                                    dc_lo(2):dc_hi(2), &
                                                    dc_lo(3):dc_hi(3))
    integer,            intent(in)  :: bxlo(3), bxhi(3)
    integer,            intent(in)  :: domlo(3), domhi(3)
    real(amrex_real),   intent(in)  :: dxinv(3) ! [1/dx^2, 1/dy^2, 1/dz^2]
    real(amrex_real),   intent(in)  :: vlo, vhi
    integer,            intent(in)  :: phase, dir ! phase kept for backwards compat
    integer,            intent(in)  :: debug_print_level

    ! Local variables
    integer :: i, j, k, m_idx, stencil_idx_start, s_idx
    integer :: len_x, len_y, len_z, expected_nval
    real(amrex_real) :: diag_val, coeff_x, coeff_y, coeff_z
    real(amrex_real) :: domain_extent, factor
    real(amrex_real) :: D_c, D_nbr, D_face
    logical :: on_dirichlet_boundary
    logical :: has_inactive_neighbor
    ! Debugging variables
    logical :: print_this_cell, near_boundary
    real(amrex_real) :: off_diag_sum, diag_ratio
    character(len=200) :: fmt_str ! Format string for printing

    ! Calculate box dimensions based on bxlo/bxhi (the valid box)
    len_x = bxhi(1) - bxlo(1) + 1
    len_y = bxhi(2) - bxlo(2) + 1
    len_z = bxhi(3) - bxlo(3) + 1
    expected_nval = len_x * len_y * len_z

    ! Pre-calculate inverse grid spacing squared
    coeff_x = dxinv(1)
    coeff_y = dxinv(2)
    coeff_z = dxinv(3)

    ! Check consistency
    if (expected_nval > 0 .and. expected_nval /= nval) then
       call amrex_abort("tortuosity_fillmtx: nval mismatch.")
    end if
    if (nval <= 0) return ! Nothing to do for empty box

    ! Loop over the valid box defined by bxlo/bxhi
    do k = bxlo(3), bxhi(3)
      do j = bxlo(2), bxhi(2)
        do i = bxlo(1), bxhi(1)

          ! Calculate indices
          m_idx = (i - bxlo(1)) + (j - bxlo(2)) * len_x + (k - bxlo(3)) * len_x * len_y + 1
          stencil_idx_start = nstencil * (m_idx - 1)

          ! Initialize debugging flags for this cell
          print_this_cell = .false.
          near_boundary = .false.
          has_inactive_neighbor = .false.

          ! --- Check if the cell is part of the simulation ---
          ! Cell must be marked active in the percolating mask.
          if ( active_mask(i,j,k) == cell_inactive ) then
              ! --- Apply Explicit Decoupling for INACTIVE cells ---
              a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
              a(stencil_idx_start + istn_c) = 1.0_amrex_real
              rhs(m_idx)  = 0.0_amrex_real
              xinit(m_idx) = 0.0_amrex_real
              cycle ! Skip to next (i,j,k)
          end if

          ! --- If we reach here, cell (i,j,k) is ACTIVE ---
          D_c = diff_coeff(i,j,k)
          diag_val = 0.0_amrex_real
          a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real

          ! --- Assemble stencil using harmonic mean face coefficients ---
          ! For each face: D_face = 2*D_c*D_nbr / (D_c + D_nbr)

          ! -X face
          if ( active_mask(i-1,j,k) == cell_active ) then
             D_nbr = diff_coeff(i-1,j,k)
             D_face = 2.0_amrex_real * D_c * D_nbr / (D_c + D_nbr)
             a(stencil_idx_start + istn_mx) = -D_face * coeff_x
             diag_val = diag_val + D_face * coeff_x
          else
             has_inactive_neighbor = .true.
          end if

          ! +X face
          if ( active_mask(i+1,j,k) == cell_active ) then
             D_nbr = diff_coeff(i+1,j,k)
             D_face = 2.0_amrex_real * D_c * D_nbr / (D_c + D_nbr)
             a(stencil_idx_start + istn_px) = -D_face * coeff_x
             diag_val = diag_val + D_face * coeff_x
          else
             has_inactive_neighbor = .true.
          end if

          ! -Y face
          if ( active_mask(i,j-1,k) == cell_active ) then
             D_nbr = diff_coeff(i,j-1,k)
             D_face = 2.0_amrex_real * D_c * D_nbr / (D_c + D_nbr)
             a(stencil_idx_start + istn_my) = -D_face * coeff_y
             diag_val = diag_val + D_face * coeff_y
          else
             has_inactive_neighbor = .true.
          end if

          ! +Y face
          if ( active_mask(i,j+1,k) == cell_active ) then
             D_nbr = diff_coeff(i,j+1,k)
             D_face = 2.0_amrex_real * D_c * D_nbr / (D_c + D_nbr)
             a(stencil_idx_start + istn_py) = -D_face * coeff_y
             diag_val = diag_val + D_face * coeff_y
          else
             has_inactive_neighbor = .true.
          end if

          ! -Z face
          if ( active_mask(i,j,k-1) == cell_active ) then
             D_nbr = diff_coeff(i,j,k-1)
             D_face = 2.0_amrex_real * D_c * D_nbr / (D_c + D_nbr)
             a(stencil_idx_start + istn_mz) = -D_face * coeff_z
             diag_val = diag_val + D_face * coeff_z
          else
             has_inactive_neighbor = .true.
          end if

          ! +Z face
          if ( active_mask(i,j,k+1) == cell_active ) then
             D_nbr = diff_coeff(i,j,k+1)
             D_face = 2.0_amrex_real * D_c * D_nbr / (D_c + D_nbr)
             a(stencil_idx_start + istn_pz) = -D_face * coeff_z
             diag_val = diag_val + D_face * coeff_z
          else
             has_inactive_neighbor = .true.
          end if

          ! Set the diagonal entry
          a(stencil_idx_start + istn_c) = diag_val

          ! Check for zero diagonal in an active cell
          if ( abs(diag_val) < small_real ) then
             write(*,'(A,3I5)') "WARNING: Zero diagonal in ACTIVE cell at (i,j,k)=", i,j,k
             a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
             a(stencil_idx_start + istn_c) = 1.0_amrex_real
             rhs(m_idx)  = 0.0_amrex_real
             xinit(m_idx) = 0.0_amrex_real
             cycle
          else
             rhs(m_idx) = 0.0_amrex_real
          endif


          ! --- Overwrite stencil for Domain Boundaries Perpendicular to Flow (Dirichlet) ---
          on_dirichlet_boundary = .false.
          if ( dir == direction_x ) then
              if ( i == domlo(1) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vlo
                  on_dirichlet_boundary = .true.
              else if ( i == domhi(1) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vhi
                  on_dirichlet_boundary = .true.
              end if
          else if ( dir == direction_y ) then
              if ( j == domlo(2) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vlo
                  on_dirichlet_boundary = .true.
              else if ( j == domhi(2) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vhi
                  on_dirichlet_boundary = .true.
              end if
          else if ( dir == direction_z ) then
              if ( k == domlo(3) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vlo
                  on_dirichlet_boundary = .true.
              else if ( k == domhi(3) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vhi
                  on_dirichlet_boundary = .true.
              end if
          end if ! End of Dirichlet BC overwrite

          ! --- Calculate Initial Guess (Only for ACTIVE cells) ---
          if ( abs(a(stencil_idx_start + istn_c) - 1.0_amrex_real) > small_real .or. on_dirichlet_boundary ) then
             if ( dir == direction_x ) then
                 domain_extent = domhi(1) - domlo(1)
                 if (abs(domain_extent) < small_real) then
                   factor = 0.0_amrex_real
                 else
                   factor = 1.0_amrex_real / domain_extent
                 end if
                 xinit(m_idx) = vlo + (vhi - vlo) * (i - domlo(1)) * factor
             else if ( dir == direction_y ) then
                 domain_extent = domhi(2) - domlo(2)
                 if (abs(domain_extent) < small_real) then
                   factor = 0.0_amrex_real
                 else
                   factor = 1.0_amrex_real / domain_extent
                 end if
                 xinit(m_idx) = vlo + (vhi - vlo) * (j - domlo(2)) * factor
             else if ( dir == direction_z ) then
                 domain_extent = domhi(3) - domlo(3)
                 if (abs(domain_extent) < small_real) then
                   factor = 0.0_amrex_real
                 else
                   factor = 1.0_amrex_real / domain_extent
                 end if
                 xinit(m_idx) = vlo + (vhi - vlo) * (k - domlo(3)) * factor
             else ! Should not happen
                 xinit(m_idx) = 0.5_amrex_real * (vlo + vhi)
             end if
          endif ! End check for calculating xinit ramp

          ! --- ** Debug Printing Section (Activated if debug_print_level >= 3) ** ---
          if (debug_print_level >= 3) then
              near_boundary = (i <= domlo(1)+1 .or. i >= domhi(1)-1 .or. &
                               j <= domlo(2)+1 .or. j >= domhi(2)-1 .or. &
                               k <= domlo(3)+1 .or. k >= domhi(3)-1)

              print_this_cell = near_boundary .or. has_inactive_neighbor

              if (print_this_cell) then
                  off_diag_sum = 0.0_amrex_real
                  do s_idx = 1, nstencil-1
                      off_diag_sum = off_diag_sum + abs(a(stencil_idx_start + s_idx))
                  end do

                  diag_val = a(stencil_idx_start + istn_c)
                  if (abs(off_diag_sum) < small_real) then
                      if (abs(diag_val) < small_real) then
                          diag_ratio = 1.0_amrex_real
                      else
                          diag_ratio = 1.0e+30_amrex_real
                      end if
                  else
                      diag_ratio = abs(diag_val) / off_diag_sum
                  end if

                  write(*,'(A,3I5,A,L1,A,L1,A,L1,A,ES12.4)') "DEBUG Stencil at (", i, j, k, ")", &
                      " Active=", .true., &
                      " Dirichlet=", on_dirichlet_boundary, &
                      " Interface=", has_inactive_neighbor, &
                      " D_c=", D_c
                  write(*,'(A,ES12.4)')   "  RHS =", rhs(m_idx)
                  write(*,'(A,7(ES12.4,1X))') "  Stencil (C, -X,+X, -Y,+Y, -Z,+Z) =" , &
                                             a(stencil_idx_start + istn_c), a(stencil_idx_start + istn_mx), &
                                             a(stencil_idx_start + istn_px), a(stencil_idx_start + istn_my), &
                                             a(stencil_idx_start + istn_py), a(stencil_idx_start + istn_mz), &
                                             a(stencil_idx_start + istn_pz)
                  write(*,'(A,ES12.4, A,ES12.4)') "  Diag Dominance Ratio (|Aii|/Sum|Aij|) =", diag_ratio, &
                                                 " (OffDiagSum =", off_diag_sum, ")"
              end if !(print_this_cell)
          end if !(debug_print_level >= 3)
          ! --- ** End Debug Printing Section ** ---

        end do ! i
      end do ! j
    end do ! k

  end subroutine tortuosity_fillmtx

end module tortuosity_poisson_3d_module
