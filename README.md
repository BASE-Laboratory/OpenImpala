
# OpenImpala
<p align="center">
  <img width="1729" height="910" alt="Banner" src="https://github.com/user-attachments/assets/8af54b12-c3c2-4c9a-8416-d2eacb626a9d" />
</p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BASE-Laboratory/OpenImpala/blob/master/tutorials/01_hello_openimpala.ipynb)
[![PyPI](https://img.shields.io/pypi/v/openimpala)](https://pypi.org/project/openimpala/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.softx.2021.100729-blue)](https://doi.org/10.1016/j.softx.2021.100729)
[![GitHub release](https://img.shields.io/github/v/release/BASE-Laboratory/OpenImpala)](https://github.com/BASE-Laboratory/OpenImpala/releases/latest)
[![Build and Test](https://github.com/BASE-Laboratory/OpenImpala/actions/workflows/build-test.yml/badge.svg?branch=master)](https://github.com/BASE-Laboratory/OpenImpala/actions/workflows/build-test.yml)
[![codecov](https://codecov.io/gh/BASE-Laboratory/OpenImpala/branch/master/graph/badge.svg)](https://codecov.io/gh/BASE-Laboratory/OpenImpala)

**OpenImpala** is a high-performance framework for computing effective
transport properties — tortuosity, effective diffusivity tensors, and
effective conductivity — directly from 3D voxel images of porous media (X-ray
CT, FIB-SEM, synthetic microstructures). It solves the governing PDEs on the
Cartesian voxel grid using finite differences, bypassing mesh generation, and
scales across MPI ranks and CUDA GPUs via [AMReX](https://github.com/AMReX-Codes/amrex)
and [HYPRE](https://github.com/hypre-space/hypre).

Outputs parameterise continuum-scale models such as
[PyBaMM](https://github.com/pybamm-team/PyBaMM).

📖 **Documentation:** <https://base-laboratory.github.io/OpenImpala/>
📓 **Tutorials:** [`tutorials/`](tutorials/) (runnable on Google Colab)

## Quick example

```python
import numpy as np
import openimpala as oi

image = np.zeros((64, 64, 64), dtype=np.int32)
image[:, :, 16:48] = 1   # solid slab through the middle

with oi.Session():
    vf  = oi.volume_fraction(image, phase=0)
    tau = oi.tortuosity(image, phase=0, direction="z", solver="mlmg")
    print(f"Volume fraction: {vf.value:.4f}")
    print(f"Tortuosity:      {tau.value:.4f}")
```

## Install

### Python (recommended)

```bash
pip install openimpala            # CPU + optional CuPy GPU acceleration
pip install openimpala-cuda       # compiled HYPRE/AMReX CUDA wheel (Linux x86_64)
```

OpenImpala uses MPI for distributed parallelism — install an MPI runtime
(`libopenmpi-dev`, `openmpi`, `brew install open-mpi`, or
`conda install -c conda-forge openmpi`) before `pip install`. See
[Getting Started](https://base-laboratory.github.io/OpenImpala/getting-started.html)
for full details.

### Container (HPC)

Pre-built Apptainer/Singularity images are attached to each
[GitHub Release](https://github.com/BASE-Laboratory/OpenImpala/releases):

```bash
apptainer exec -B "$(pwd):/data" openimpala-vX.Y.Z.sif \
    /usr/local/bin/Diffusion /data/inputs
```

For batch SLURM scripts, see
[HPC Usage](https://base-laboratory.github.io/OpenImpala/user-guide/hpc.html).

### Build from source

See [CONTRIBUTING.md](CONTRIBUTING.md) for the native and containerised
developer build, code style, and test workflow.

## Features

- Steady-state diffusion / conduction on segmented 3D voxel images
- Tortuosity factor, full 3×3 effective diffusivity tensor, multi-phase transport
- Microstructural metrics: volume fraction, percolation, particle size, specific surface area
- TIFF / HDF5 / RAW / DAT image input; JSON output compatible with BPX / BattINFO
- Solvers: HYPRE (PCG, FlexGMRES, BiCGSTAB; SMG / PFMG preconditioners) and AMReX MLMG (matrix-free, GPU-native)
- MPI + OpenMP + CUDA parallelism — scales from a laptop to multi-node HPC

## Citation

If you use OpenImpala in published work, please cite:

```bibtex
@article{LeHoux2021OpenImpala,
  title   = {{OpenImpala}: {OPEN} source {IMage} based {PArallisable} {Linear} {Algebra} solver},
  author  = {Le Houx, James and Kramer, Denis},
  year    = {2021},
  journal = {SoftwareX},
  volume  = {15},
  pages   = {100729},
  doi     = {10.1016/j.softx.2021.100729},
}
```

If you use the homogenisation-based effective diffusivity workflow,
additionally cite Le Houx et al., *Transport in Porous Media* **150**, 71–88
(2023), [doi:10.1007/s11242-023-01993-7](https://doi.org/10.1007/s11242-023-01993-7).

## License

BSD 3-Clause. See [LICENSE](LICENSE).

## Acknowledgements

This work was financially supported by the EPSRC Centre for Doctoral Training
in Energy Storage and its Applications [EP/R021295/1]; the Ada Lovelace Centre
(STFC) project CANVAS-NXtomo; the EPSRC prosperity partnership with Imperial
College, INFUSE [EP/V038044/1]; the Rutherford Appleton Laboratory; the
Faraday Institution Emerging Leader Fellowship [FIELF001]; and Research
England's *Expanding Excellence in England* grant at the University of
Greenwich via the M34Impact programme. We acknowledge the use of the IRIDIS
HPC facility, Diamond Light Source's Wilson cluster, STFC SCARF, and the
University of Greenwich M34Impact cluster, and thank the developers of AMReX,
HYPRE, libtiff, and HDF5.

## Contact

Issues and feature requests: <https://github.com/BASE-Laboratory/OpenImpala/issues>.
Questions: [GitHub Discussions](https://github.com/BASE-Laboratory/OpenImpala/discussions).
