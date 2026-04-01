# Scientific Verification & Validation (V&V) Suite

This directory contains scripts that validate OpenImpala's physics solvers
against known analytical solutions and reference datasets. It complements —
but is distinct from — the regression tests in the parent `tests/` directory.

## Regression Tests vs. V&V

| Aspect | Regression Tests (`ctest`) | V&V Suite (`tests/validation/`) |
|--------|---------------------------|--------------------------------|
| **Purpose** | Detect *unintended changes* in output | Prove *physical correctness* of output |
| **Reference** | Previous code output (frozen values) | Analytical theory / experimental data |
| **Passes when** | Output matches stored baseline exactly | Output falls within mathematical bounds |
| **Fails when** | Code change alters any result | Result violates a physical law |
| **Example** | "Tortuosity on this TIFF is still 1.2345" | "D_eff lies between Reuss and Voigt bounds" |

Regression tests catch bugs but cannot prove correctness — a bug that has
been present since the baseline was set will pass regression. V&V tests
prove that the solver respects fundamental physics, independent of any
previous output.

## Scripts

### `analytical_bounds_vv.py` — Effective Diffusivity Bounds

Validates that OpenImpala's computed effective diffusivity respects the
Wiener and Hashin-Shtrikman bounds from composite materials theory.

**What it does:**

1. Generates synthetic 3D microstructures (parallel layers, random mixtures)
   at varying volume fractions (0.1 to 0.9).
2. Runs the OpenImpala tortuosity solver on each structure.
3. Converts tortuosity to effective diffusivity: D_eff = VF / τ.
4. Checks that every result falls within the analytical bounds.
5. Produces a publication-ready plot (`validation_bounds.png`).

**How to run:**

```bash
# From the repository root
python tests/validation/analytical_bounds_vv.py

# With custom grid size and output path
python tests/validation/analytical_bounds_vv.py --grid-size 48 --output my_plot.png
```

**Exit code:** 0 if all bounds are satisfied, 1 if any violation is detected.

### `fetch_canonical_data.py` — Experimental Dataset Validation

Downloads a canonical 3D TIFF dataset and validates the solver output against
stored reference values (porosity and tortuosity).

**What it does:**

1. Downloads `SampleData_2Phase_stack_3d_1bit.tif` from the OpenImpala
   GitHub repository (or copies from the local `data/` directory if offline).
2. Writes `canonical_reference.json` with the expected experimental values.
3. Runs the solver and checks that computed porosity and tortuosity are
   within 5% of the reference values.
4. Writes `validation_results.json` with the full solver output.

**How to run:**

```bash
# From the repository root
python tests/validation/fetch_canonical_data.py

# With custom data directory
python tests/validation/fetch_canonical_data.py --data-dir /tmp/vv_data
```

**Exit code:** 0 if within tolerance, 1 if any value exceeds the threshold.

### `sphere_packing_vv.py` — Sphere Packing HS Bound Validation

Validates that OpenImpala's tortuosity solver respects the Hashin-Shtrikman
upper bound on physically realistic isotropic microstructures generated from
random overlapping sphere packings.

**What it does:**

1. Generates random overlapping sphere packings at varying solid fractions
   (porosities from ~0.25 to ~0.85).
2. Checks percolation of the pore phase in the solve direction.
3. Runs the OpenImpala tortuosity solver on each percolating structure.
4. Converts tortuosity to effective diffusivity: D_eff = φ / τ.
5. Checks that every result falls below the HS upper bound: HS⁺ = 2φ/(3−φ).
6. Produces a validation plot (`sphere_packing_vv.png`).

**How to run:**

```bash
# From the repository root
python tests/validation/sphere_packing_vv.py

# With custom grid size and output path
python tests/validation/sphere_packing_vv.py --grid-size 48 --output my_plot.png
```

**Exit code:** 0 if all results are within the HS upper bound, 1 if any
violation is detected.

## The Hashin-Shtrikman Bounds: Physical Significance

The Hashin-Shtrikman (HS) bounds are the **tightest possible bounds** on the
effective transport properties of a composite material when only the volume
fractions and phase properties are known (no geometric information).

For a two-phase composite with diffusivities D₀ > D₁ > 0 and volume
fraction φ of phase 1:

```
HS⁻ = D₁ + (1-φ) / ( 1/(D₀-D₁) + φ/(3·D₁) )

HS⁺ = D₀ + φ / ( 1/(D₁-D₀) + (1-φ)/(3·D₀) )
```

### Why these bounds matter

1. **Any valid solver must produce results within HS bounds.** If a computed
   D_eff falls outside [HS⁻, HS⁺], either the solver has a bug or the
   problem setup is incorrect. This is a necessary (not sufficient) condition
   for correctness.

2. **HS bounds are tighter than Wiener bounds.** The Wiener bounds (Voigt
   upper = arithmetic mean, Reuss lower = harmonic mean) use only volume
   fractions. HS bounds additionally assume statistical isotropy, which
   tightens the feasible region significantly.

3. **HS bounds are realised by specific geometries.** The upper bound
   corresponds to the Hashin coated-sphere assemblage where the
   high-diffusivity phase forms the matrix and the low-diffusivity phase
   forms spherical inclusions. The lower bound reverses the roles. This
   means the bounds are *attainable* — they are not conservative estimates.

### For contributors adding new physics

If you implement a new transport solver (e.g., for thermal conductivity,
electrical conductivity, or permeability), the HS bounds provide an
immediate V&V framework:

- **Step 1:** Implement the bound functions for your transport equation.
  For conductivity and diffusivity, the same HS formulas apply directly
  (the governing equations are mathematically identical).

- **Step 2:** Generate test structures at several volume fractions.

- **Step 3:** Verify that your solver output falls within [HS⁻, HS⁺].

If it does, you have strong evidence of correctness. If it does not,
investigate before merging.

## Directory Structure

```
tests/validation/
├── README.md                      ← this file
├── analytical_bounds_vv.py        ← bounds validation (layered/random structures)
├── sphere_packing_vv.py           ← bounds validation (sphere packings)
├── fetch_canonical_data.py        ← dataset fetcher + reference validation
└── data/                          ← downloaded datasets and reference JSON
    ├── SampleData_2Phase_*.tif    ← canonical TIFF (downloaded on first run)
    ├── canonical_reference.json   ← expected experimental values
    └── validation_results.json    ← solver output from last run
```

## References

1. Z. Hashin & S. Shtrikman, "A variational approach to the theory of the
   effective magnetic permeability of multiphase materials", J. Applied
   Physics 33(10), 3125–3131 (1962).
   [doi:10.1063/1.1728579](https://doi.org/10.1063/1.1728579)

2. S. Torquato, *Random Heterogeneous Materials: Microstructure and
   Macroscopic Properties*, Springer (2002). Chapters 17–21 cover
   effective property bounds comprehensively.

3. G. W. Milton, *The Theory of Composites*, Cambridge University Press
   (2002). The definitive reference on composite bounds theory.
