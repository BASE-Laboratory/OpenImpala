module effdiff_fillmtx_module

  use amrex_fort_module, only : amrex_real, amrex_spacedim
  use amrex_error_module, only : amrex_abort
  implicit none

  private ! Default module visibility to private
  public :: effdiff_fillmtx ! Make only the subroutine public

  ! --- Constants ---
  integer, parameter :: DIR_X = 0
  integer, parameter :: DIR_Y = 1
  integer, parameter :: DIR_Z = 2

  ! active_mask values (consistent with C++)
  integer, parameter :: CELL_INACTIVE = 0 ! Solid, D=0
  integer, parameter :: CELL_ACTIVE   = 1 ! Pore/conducting phase

  ! Stencil entry indices (0-based)
  integer, parameter :: STN_C  = 0 ! Center
  integer, parameter :: STN_MX = 1 ! -X (West)
  integer, parameter :: STN_PX = 2 ! +X (East)
  integer, parameter :: STN_MY = 3 ! -Y (South)
  integer, parameter :: STN_PY = 4 ! +Y (North)
  integer, parameter :: STN_MZ = 5 ! -Z (Bottom)
  integer, parameter :: STN_PZ = 6 ! +Z (Top)
  integer, parameter :: NSTENCIL = 7

  real(amrex_real), parameter :: SMALL_REAL = 1.0e-15_amrex_real
  real(amrex_real), parameter :: ONE = 1.0_amrex_real
  real(amrex_real), parameter :: ZERO = 0.0_amrex_real
  real(amrex_real), parameter :: HALF = 0.5_amrex_real
  real(amrex_real), parameter :: TWO = 2.0_amrex_real


contains

  ! Subroutine to fill HYPRE matrix (A) and RHS (b) for the cell problem:
  ! div(D grad(chi_k)) = -div(D e_k)
  ! where D is the spatially varying diffusion coefficient from diff_coeff.
  ! Uses harmonic mean for face coefficients between cells.
  subroutine effdiff_fillmtx(a_out, rhs_out, xinit_out, &
                             npts_valid, &
                             active_mask_ptr, mask_lo, mask_hi, &
                             diff_coeff_ptr, dc_lo, dc_hi, &
                             valid_bx_lo, valid_bx_hi, &
                             domain_lo, domain_hi, &
                             cell_sizes_in, & ! dx, dy, dz
                             dir_k_in, &
                             verbose_level_in) bind(c)

    ! --- Argument Declarations ---
    integer, intent(in) :: npts_valid
    real(amrex_real), intent(out) :: a_out(0:npts_valid*NSTENCIL-1)
    real(amrex_real), intent(out) :: rhs_out(npts_valid)
    real(amrex_real), intent(out) :: xinit_out(npts_valid)

    integer, intent(in) :: mask_lo(3), mask_hi(3)
    integer, intent(in) :: active_mask_ptr(mask_lo(1):mask_hi(1), mask_lo(2):mask_hi(2), mask_lo(3):mask_hi(3))

    integer, intent(in) :: dc_lo(3), dc_hi(3)
    real(amrex_real), intent(in) :: diff_coeff_ptr(dc_lo(1):dc_hi(1), dc_lo(2):dc_hi(2), dc_lo(3):dc_hi(3))

    integer, intent(in) :: valid_bx_lo(3), valid_bx_hi(3)
    integer, intent(in) :: domain_lo(3), domain_hi(3)

    real(amrex_real), intent(in) :: cell_sizes_in(3) ! dx, dy, dz
    integer, intent(in) :: dir_k_in            ! 0 for X, 1 for Y, 2 for Z
    integer, intent(in) :: verbose_level_in

    ! --- Local Variables ---
    integer :: i, j, k, m_idx, stencil_idx_start, s_idx
    integer :: len_x_valid, len_y_valid
    real(amrex_real) :: dx, dy, dz
    real(amrex_real) :: inv_dx2, inv_dy2, inv_dz2
    real(amrex_real) :: inv_2dx, inv_2dy, inv_2dz

    real(amrex_real) :: diag_val
    real(amrex_real) :: D_c, D_nbr, D_face
    real(amrex_real) :: rhs_term_div_De
    real(amrex_real) :: flux_bc_contrib_rhs

    ! --- Initialization ---
    if (npts_valid <= 0) return

    dx = cell_sizes_in(1)
    dy = cell_sizes_in(2)
    dz = cell_sizes_in(3)

    if (dx <= SMALL_REAL .or. dy <= SMALL_REAL .or. dz <= SMALL_REAL) then
      call amrex_abort("effdiff_fillmtx: cell_sizes (dx, dy, dz) must be positive.")
    end if

    inv_dx2 = ONE / (dx * dx)
    inv_dy2 = ONE / (dy * dy)
    inv_dz2 = ONE / (dz * dz)
    inv_2dx = ONE / (TWO * dx)
    inv_2dy = ONE / (TWO * dy)
    inv_2dz = ONE / (TWO * dz)

    len_x_valid = valid_bx_hi(1) - valid_bx_lo(1) + 1
    len_y_valid = valid_bx_hi(2) - valid_bx_lo(2) + 1

    m_idx = 0

    do k = valid_bx_lo(3), valid_bx_hi(3)
      do j = valid_bx_lo(2), valid_bx_hi(2)
        do i = valid_bx_lo(1), valid_bx_hi(1)
          m_idx = m_idx + 1
          stencil_idx_start = NSTENCIL * (m_idx - 1)

          ! Initialize
          a_out(stencil_idx_start : stencil_idx_start + NSTENCIL - 1) = ZERO
          rhs_out(m_idx)  = ZERO
          xinit_out(m_idx) = ZERO

          if (active_mask_ptr(i, j, k) == CELL_INACTIVE) then
            ! Solid cell: decouple
            a_out(stencil_idx_start + STN_C) = ONE
            cycle
          end if

          ! --- Active cell: D_c from diff_coeff ---
          D_c = diff_coeff_ptr(i, j, k)
          diag_val = ZERO
          flux_bc_contrib_rhs = ZERO

          ! === LHS: div(D grad(chi_k)) using harmonic mean at faces ===

          ! -X face (West)
          if (active_mask_ptr(i-1, j, k) == CELL_ACTIVE) then
            D_nbr = diff_coeff_ptr(i-1, j, k)
            D_face = TWO * D_c * D_nbr / (D_c + D_nbr)
            a_out(stencil_idx_start + STN_MX) = -D_face * inv_dx2
            diag_val = diag_val + D_face * inv_dx2
          else
            ! Internal Neumann BC: n_hat=(-1,0,0)
            ! D * dchi/dn = -D * (e_k . n_hat) => dchi/dx_face = (e_k)_x
            diag_val = diag_val + D_c * inv_dx2
            if (dir_k_in == DIR_X) then
              flux_bc_contrib_rhs = flux_bc_contrib_rhs + D_c * (ONE/dx)
            end if
          end if

          ! +X face (East)
          if (active_mask_ptr(i+1, j, k) == CELL_ACTIVE) then
            D_nbr = diff_coeff_ptr(i+1, j, k)
            D_face = TWO * D_c * D_nbr / (D_c + D_nbr)
            a_out(stencil_idx_start + STN_PX) = -D_face * inv_dx2
            diag_val = diag_val + D_face * inv_dx2
          else
            diag_val = diag_val + D_c * inv_dx2
            if (dir_k_in == DIR_X) then
              flux_bc_contrib_rhs = flux_bc_contrib_rhs - D_c * (ONE/dx)
            end if
          end if

          ! -Y face (South)
          if (active_mask_ptr(i, j-1, k) == CELL_ACTIVE) then
            D_nbr = diff_coeff_ptr(i, j-1, k)
            D_face = TWO * D_c * D_nbr / (D_c + D_nbr)
            a_out(stencil_idx_start + STN_MY) = -D_face * inv_dy2
            diag_val = diag_val + D_face * inv_dy2
          else
            diag_val = diag_val + D_c * inv_dy2
            if (dir_k_in == DIR_Y) then
              flux_bc_contrib_rhs = flux_bc_contrib_rhs + D_c * (ONE/dy)
            end if
          end if

          ! +Y face (North)
          if (active_mask_ptr(i, j+1, k) == CELL_ACTIVE) then
            D_nbr = diff_coeff_ptr(i, j+1, k)
            D_face = TWO * D_c * D_nbr / (D_c + D_nbr)
            a_out(stencil_idx_start + STN_PY) = -D_face * inv_dy2
            diag_val = diag_val + D_face * inv_dy2
          else
            diag_val = diag_val + D_c * inv_dy2
            if (dir_k_in == DIR_Y) then
              flux_bc_contrib_rhs = flux_bc_contrib_rhs - D_c * (ONE/dy)
            end if
          end if

          ! -Z face (Bottom)
          if (AMREX_SPACEDIM == 3) then
            if (active_mask_ptr(i, j, k-1) == CELL_ACTIVE) then
              D_nbr = diff_coeff_ptr(i, j, k-1)
              D_face = TWO * D_c * D_nbr / (D_c + D_nbr)
              a_out(stencil_idx_start + STN_MZ) = -D_face * inv_dz2
              diag_val = diag_val + D_face * inv_dz2
            else
              diag_val = diag_val + D_c * inv_dz2
              if (dir_k_in == DIR_Z) then
                flux_bc_contrib_rhs = flux_bc_contrib_rhs + D_c * (ONE/dz)
              end if
            end if

            ! +Z face (Top)
            if (active_mask_ptr(i, j, k+1) == CELL_ACTIVE) then
              D_nbr = diff_coeff_ptr(i, j, k+1)
              D_face = TWO * D_c * D_nbr / (D_c + D_nbr)
              a_out(stencil_idx_start + STN_PZ) = -D_face * inv_dz2
              diag_val = diag_val + D_face * inv_dz2
            else
              diag_val = diag_val + D_c * inv_dz2
              if (dir_k_in == DIR_Z) then
                flux_bc_contrib_rhs = flux_bc_contrib_rhs - D_c * (ONE/dz)
              end if
            end if
          end if ! AMREX_SPACEDIM == 3

          a_out(stencil_idx_start + STN_C) = diag_val

          ! === RHS: -div(D e_k) using central differences on D ===
          rhs_term_div_De = ZERO
          if (dir_k_in == DIR_X) then
            rhs_term_div_De = -(diff_coeff_ptr(i+1,j,k) - diff_coeff_ptr(i-1,j,k)) * inv_2dx
          else if (dir_k_in == DIR_Y) then
            rhs_term_div_De = -(diff_coeff_ptr(i,j+1,k) - diff_coeff_ptr(i,j-1,k)) * inv_2dy
          else if (dir_k_in == DIR_Z .and. AMREX_SPACEDIM == 3) then
            rhs_term_div_De = -(diff_coeff_ptr(i,j,k+1) - diff_coeff_ptr(i,j,k-1)) * inv_2dz
          end if

          rhs_out(m_idx) = rhs_term_div_De + flux_bc_contrib_rhs

          ! Safety: decouple cells with near-zero diagonal
          if (abs(diag_val) < SMALL_REAL) then
             if (verbose_level_in > 0) then
                 write(*,'(A,3I5,A,ES12.4)') "effdiff_fillmtx WARNING: Near-zero diagonal in ACTIVE cell (", &
                      i,j,k, "), diag_val=", diag_val, " Decoupling."
             end if
             a_out(stencil_idx_start : stencil_idx_start + NSTENCIL - 1) = ZERO
             a_out(stencil_idx_start + STN_C) = ONE
             rhs_out(m_idx) = ZERO
             xinit_out(m_idx) = ZERO
          end if

          if (verbose_level_in >= 3) then
            write(*,'(A,3I5,A,I2,A,ES12.4)') "DEBUG effdiff_fillmtx: Cell (",i,j,k,") dir_k=", dir_k_in, &
                                               " D_c=", D_c
            write(*,'(A,7ES12.4)') "  Stencil A: ", (a_out(stencil_idx_start+s_idx), s_idx=0,NSTENCIL-1)
            write(*,'(A,ES12.4, A,ES12.4, A,ES12.4)') "  RHS terms: div_De=", rhs_term_div_De, &
                                                   " flux_bc=", flux_bc_contrib_rhs, &
                                                   " Total_RHS=", rhs_out(m_idx)
          end if

        end do ! i
      end do ! j
    end do ! k

    if (m_idx /= npts_valid) then
      call amrex_abort("effdiff_fillmtx: m_idx /= npts_valid. Indexing error.")
    end if

  end subroutine effdiff_fillmtx

end module effdiff_fillmtx_module
