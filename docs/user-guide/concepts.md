# Concepts

## What OpenImpala computes

OpenImpala takes a **segmented 3D voxel image** (where each voxel is labelled
with a phase ID) and computes **effective transport properties** by solving
partial differential equations directly on the voxel grid.

### Phase data

Images are segmented into integer phase IDs stored in an AMReX `iMultiFab`.
Typically:

- **Phase 0** = pore / void
- **Phase 1** = solid matrix

This is configurable via the `phase` parameter. Multi-phase transport is
supported: each phase can be assigned a different transport coefficient.

### Volume fraction

The simplest metric: the fraction of voxels belonging to a given phase.

$$\varepsilon = \frac{N_{\text{phase}}}{N_{\text{total}}}$$

### Percolation

Before solving transport equations, OpenImpala checks whether the target phase
forms a **connected path** from inlet to outlet using a GPU-accelerated
flood-fill algorithm. If the phase does not percolate, transport is zero.

### Tortuosity

Tortuosity quantifies how much a winding pore structure impedes transport
compared to a straight channel. OpenImpala solves the steady-state diffusion
equation:

$$\nabla \cdot (D \nabla \phi) = 0$$

with Dirichlet boundary conditions at inlet ($\phi = 0$) and outlet
($\phi = 1$), and zero-flux Neumann conditions on lateral faces.

The effective diffusivity is computed from the resulting flux:

$$D_{\text{eff}} = \frac{|\text{average flux}|}{\text{cross-section area} \times |\nabla\phi_{\text{imposed}}|}$$

Tortuosity is then:

$$\tau = \frac{\varepsilon_{\text{active}}}{D_{\text{eff}}}$$

where $\varepsilon_{\text{active}}$ is the volume fraction of the percolating
(connected) phase.

For a uniform medium on an $N$-cell grid, the discrete solution gives
$D_{\text{eff}} = N/(N-1)$, so $\tau = (N-1)/N$.

### Effective diffusivity tensor

For anisotropic microstructures, the full effective diffusivity tensor
$\mathbf{D}_{\text{eff}}$ is computed by solving the **cell problem** from
homogenisation theory:

$$\nabla_\xi \cdot \left( D \nabla_\xi \chi_k \right) = -\nabla_\xi \cdot \left( D \hat{e}_k \right)$$

for corrector functions $\chi_k$ in each direction $k \in \{x, y, z\}$, with
periodic boundary conditions. The tensor components are:

$$D_{\text{eff},ij} = \frac{1}{|Y|} \int_Y D(\mathbf{x}) \left( \delta_{ij} + \frac{\partial \chi_j}{\partial x_i} \right) \, d\mathbf{x}$$

### Face coefficients

Inter-cell diffusivities use the **harmonic mean** of adjacent cell values:

$$D_{\text{face}} = \frac{2 D_L D_R}{D_L + D_R}$$

This is physically correct for resistances in series and ensures that a solid
cell ($D = 0$) adjacent to a pore cell correctly blocks transport.
