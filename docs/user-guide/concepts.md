# Mathematical Formulation

OpenImpala discretises steady-state conservation equations directly on segmented 3D voxel grids using a cell-centred conservative finite-volume (7-point stencil) scheme.

## Array and Coordinate Conventions

All 3D NumPy arrays ingested or returned by OpenImpala follow `(Nz, Ny, Nx)` memory layout (`z` = axis 0, `y` = axis 1, `x` = axis 2) to match multi-page TIFF stacks and HDF5 datasets:

- **Voxel spacing**: scalar `voxel_size=h` or anisotropic 3-tuple `voxel_size=(dz, dy, dx)`.
- **Tensor and directional ordering**: `EffectiveDiffusivityResult.tensor` has rows and columns ordered `(x, y, z)`, i.e. index `(0, 1, 2)` corresponds to `(X, Y, Z)`.

## Steady-State Flow-Through Diffusion and Tortuosity

Given a conducting domain $\Omega_p \subset [0, L_x] \times [0, L_y] \times [0, L_z]$ with local diffusivity $D(\mathbf{x}) > 0$ (and $D(\mathbf{x}) = 0$ in insulating solid cells), OpenImpala solves

$$-\nabla \cdot \bigl(D(\mathbf{x})\,\nabla \phi\bigr) = 0 \quad \text{in } \Omega_p,$$

with zero-flux Neumann conditions ($\nabla \phi \cdot \mathbf{n} = 0$) on all internal solid walls and the four lateral domain faces. On the inlet and outlet faces normal to the transport direction $d \in \{x, y, z\}$, fixed potentials $\phi_{\text{lo}} = 0$ and $\phi_{\text{hi}} = 1$ are imposed at the **physical domain boundary** ($h_d / 2$ from the outermost cell centres), contributing half-cell boundary conductance $2 D_i A_d / h_d$.

:::{note}
Because the Dirichlet potentials sit on the exterior cell faces rather than on the first and last cell centres, a uniform medium or straight parallel channel of arbitrary grid resolution $N$ yields the exact continuum values $D_{\text{eff}} / D_0 = \varepsilon$ and $\tau = 1$ to solver tolerance (prior to `v4.5.0`, cell-centred pinning introduced an $O(1/N)$ offset $\tau = (N-1)/N$).
:::

Across interior faces separating cells $i$ and $j$, normal flux continuity requires the **harmonic-mean face diffusivity**

$$D_{i+1/2} = \begin{cases} \dfrac{2 D_i D_j}{D_i + D_j}, & D_i > 0 \text{ and } D_j > 0, \\[6pt] 0, & \text{otherwise.} \end{cases}$$

From the cross-sectionally averaged steady-state flux $J_d$, the effective diffusivity and tortuosity factor are

$$D_{\text{eff},d} = \frac{|J_d|\,L_d}{A_d\,|\phi_{\text{hi}} - \phi_{\text{lo}}|}, \qquad \tau_d = \frac{\varepsilon_{\text{active}}\,D_{\text{ref}}}{D_{\text{eff},d}},$$

where for single-phase runs $D_{\text{ref}} = 1$ so $\tau_d = \varepsilon / D_{\text{eff},d}$.

## Periodic Homogenisation ($\mathbf{D}_{\text{eff}} \in \mathbb{R}^{3\times 3}$)

To capture full directional anisotropy—including off-diagonal cross-terms $D_{\text{eff},ij}$ when principal material axes are oblique to the image axes—`oi.effective_diffusivity()` solves three periodic unit-cell corrector problems on the 3-torus $Y$:

$$-\nabla \cdot \Bigl(D(\mathbf{x})\bigl(\mathbf{e}_k + \nabla \chi_k\bigr)\Bigr) = 0 \quad \text{in } Y, \qquad k \in \{x, y, z\}.$$

Using the same conservative harmonic face conductances $D_{i+1/2}^{(r)}$, the discrete cell-problem RHS at cell $\mathbf{i}$ is $(D_{\mathbf{i}+1/2}^{(k)} - D_{\mathbf{i}-1/2}^{(k)}) / h_k$, and the homogenised tensor is assembled from the periodic face fluxes:

$$D_{\text{eff},rc} = \frac{1}{N_{\text{total}}} \sum_{\mathbf{i} \in Y} D_{\mathbf{i}+1/2}^{(r)} \left(\delta_{rc} + \frac{\chi_c(\mathbf{i} + \mathbf{e}_r) - \chi_c(\mathbf{i})}{h_r}\right).$$

Because this discretisation corresponds to the inner product $\langle \mathbf{e}_r + \nabla_h \chi_r, \mathbf{e}_c + \nabla_h \chi_c \rangle_D$, the discrete tensor satisfies exact Onsager reciprocity ($D_{\text{eff},rc} = D_{\text{eff},cr}$), positive semi-definiteness, and discrete Wiener (Voigt/Reuss) bounds at every grid resolution.

## Cauchy–Crofton Specific Surface Area

Counting raw voxel faces between phase $a$ and phase $b$ yields the Manhattan surface area

$$S_V^{\text{raw}} = \frac{N_x h_y h_z + N_y h_x h_z + N_z h_x h_y}{N_{\text{total}}\,h_x h_y h_z}.$$

For smooth isotropic interfaces, axis-aligned voxel staircasing inflates surface area by $3/2$. `oi.specific_surface_area()` returns both `ssa` ($S_V^{\text{raw}}$) and `ssa_corrected` ($\frac{2}{3} S_V^{\text{raw}}$, the three-line Cauchy–Crofton stereological estimator).
