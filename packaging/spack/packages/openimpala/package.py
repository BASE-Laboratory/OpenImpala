# Copyright 2013-2026 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class Openimpala(CMakePackage, CudaPackage, ROCmPackage):
    """OpenImpala: scalable 3-D voxel finite-volume solver for diffusive
    transport, periodic homogenisation, and electrode microstructure
    characterisation built on AMReX and HYPRE."""

    homepage = "https://base-laboratory.github.io/OpenImpala/"
    url = "https://github.com/BASE-Laboratory/OpenImpala/archive/refs/tags/v5.2.0.tar.gz"
    git = "https://github.com/BASE-Laboratory/OpenImpala.git"

    maintainers("jameslehoux")
    license("BSD-3-Clause")

    version("master", branch="master")
    version("5.2.0", tag="v5.2.0", preferred=True)
    version("5.1.0", tag="v5.1.0")
    version("5.0.0", tag="v5.0.0")

    variant("mpi", default=True, description="Enable MPI domain decomposition and MPI-IO")
    variant("openmp", default=True, description="Enable OpenMP host threading")
    variant("python", default=True, description="Build pybind11 Python bindings (openimpala._core)")
    variant("tests", default=False, description="Build CTest verification suite")

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("cmake@3.20:", type="build")

    depends_on("mpi", when="+mpi")
    depends_on("hdf5+cxx+hl+mpi", when="+mpi")
    depends_on("hdf5+cxx+hl~mpi", when="~mpi")
    depends_on("libtiff")
    depends_on("nlohmann-json@3.11:")

    # AMReX 3-D with native linear solvers (float64/float32/mixed MLMG)
    depends_on("amrex@25.03:+linear_solvers+pic+tiny_profile dimensions=3")
    depends_on("amrex+mpi", when="+mpi")
    depends_on("amrex~mpi", when="~mpi")
    depends_on("amrex+openmp", when="+openmp")
    depends_on("amrex+cuda", when="+cuda")
    depends_on("amrex+rocm", when="+rocm")

    # HYPRE structured multigrid solvers
    depends_on("hypre@2.28:")
    depends_on("hypre+mpi", when="+mpi")
    depends_on("hypre~mpi", when="~mpi")
    depends_on("hypre+cuda", when="+cuda")
    depends_on("hypre+rocm", when="+rocm")

    extends("python", when="+python")
    depends_on("python@3.9:", when="+python", type=("build", "link", "run"))
    depends_on("py-pybind11@2.11:2", when="+python", type=("build", "link"))
    depends_on("py-numpy", when="+python", type=("build", "run"))
    depends_on("py-scipy@1.7:", when="+python", type=("build", "run"))

    def cmake_args(self):
        args = [
            self.define_from_variant("OPENIMPALA_PYTHON", "python"),
            self.define_from_variant("BUILD_TESTING", "tests"),
        ]
        if "+cuda" in self.spec:
            args.append(self.define("GPU_BACKEND", "CUDA"))
        elif "+rocm" in self.spec:
            args.append(self.define("GPU_BACKEND", "HIP"))
        else:
            args.append(self.define("GPU_BACKEND", "NONE"))
        return args
