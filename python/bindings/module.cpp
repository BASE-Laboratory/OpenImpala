/** @file module.cpp
 *  @brief Top-level pybind11 module definition for OpenImpala Python bindings.
 *
 *  Defines the PYBIND11_MODULE entry point, AMReX lifecycle helpers,
 *  the VoxelImage opaque handle, and delegates to per-subsystem init functions.
 */

#include <cstddef>
#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_RealBox.H>
#include <AMReX_iMultiFab.H>

#ifdef AMREX_USE_GPU
#include <AMReX_Gpu.H>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include <HYPRE_config.h>

#include "VoxelImage.H"

namespace py = pybind11;

// Forward declarations — each implemented in its own .cpp
void init_enums(py::module_& m);
void init_io(py::module_& m);
void init_props(py::module_& m);
void init_solvers(py::module_& m);
void init_config(py::module_& m);

// =========================================================================
// AMReX lifecycle — replaces amrex.initialize() / amrex.finalize()
// =========================================================================
static void init_amrex() {
    if (!amrex::Initialized()) {
        // Initialise with an empty argument list (no ParmParse input)
        int argc = 0;
        char** argv = nullptr;
        amrex::Initialize(argc, argv);
    }
}

static void finalize_amrex() {
    if (amrex::Initialized()) {
        amrex::Finalize();
    }
}

static bool amrex_initialized() {
    return amrex::Initialized();
}

// =========================================================================
// Build info — lets users verify which wheel they installed (CPU vs CUDA),
// whether TinyProfile is on, OpenMP thread count, and visible GPU devices.
// Critical for Colab users: §1a of the profiling notebook runs this first.
// =========================================================================
static py::dict build_info() {
    py::dict info;

#ifdef OPENIMPALA_USE_CUDA
    info["cuda_enabled"] = true;
#else
    info["cuda_enabled"] = false;
#endif

#ifdef OPENIMPALA_USE_HIP
    info["hip_enabled"] = true;
#else
    info["hip_enabled"] = false;
#endif

#ifdef OPENIMPALA_USE_GPU
    info["gpu_enabled"] = true;
#else
    info["gpu_enabled"] = false;
#endif

#ifdef _OPENMP
    info["openmp_enabled"] = true;
    info["openmp_max_threads"] = omp_get_max_threads();
#else
    info["openmp_enabled"] = false;
    info["openmp_max_threads"] = 1;
#endif

#ifdef AMREX_USE_MPI
    info["mpi_enabled"] = true;
#else
    info["mpi_enabled"] = false;
#endif

#ifdef AMREX_TINY_PROFILING
    info["tiny_profile"] = true;
#else
    info["tiny_profile"] = false;
#endif

#ifdef HYPRE_USING_CUDA
    info["hypre_cuda"] = true;
#else
    info["hypre_cuda"] = false;
#endif

#ifdef HYPRE_USING_HIP
    info["hypre_hip"] = true;
#else
    info["hypre_hip"] = false;
#endif

    // Runtime GPU device count — only meaningful when the wheel has GPU support
    // AND AMReX has been initialised (otherwise the query can segfault).
#ifdef AMREX_USE_GPU
    if (amrex::Initialized()) {
        info["gpu_device_count"] = amrex::Gpu::Device::numDevicesUsed();
    } else {
        info["gpu_device_count"] = -1; // unknown until init_amrex() called
    }
#else
    info["gpu_device_count"] = 0;
#endif

    return info;
}

// =========================================================================
// NumPy → VoxelImage factory
// =========================================================================
static std::shared_ptr<OpenImpala::VoxelImage>
voxelimage_from_numpy(py::array_t<int32_t, py::array::c_style | py::array::forcecast> arr,
                      int max_grid_size) {
    py::buffer_info buf = arr.request();

    if (buf.ndim != 3) {
        throw std::runtime_error("Input must be a 3-D NumPy array, got ndim=" +
                                 std::to_string(buf.ndim));
    }

    // NumPy C-contiguous arrays are shaped (Z, Y, X)
    int nz = static_cast<int>(buf.shape[0]);
    int ny = static_cast<int>(buf.shape[1]);
    int nx = static_cast<int>(buf.shape[2]);

    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(nx - 1, ny - 1, nz - 1));
    amrex::RealBox rb({0.0, 0.0, 0.0},
                      {static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz)});
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{0, 0, 0};

    auto img = std::make_shared<OpenImpala::VoxelImage>();
    img->geom.define(domain, &rb, amrex::CoordSys::cartesian, is_periodic.data());
    img->ba.define(domain);
    img->ba.maxSize(max_grid_size);
    img->dm.define(img->ba);
    img->mf = std::make_shared<amrex::iMultiFab>(img->ba, img->dm, 1, 1);

    const auto* host_ptr = static_cast<const int32_t*>(buf.ptr);
    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz);

    // Stage the NumPy data in a buffer the kernel below can dereference safely.
    // On CPU builds this is just the host pointer (no copy). On CUDA builds the
    // iMultiFab data lives in device memory, so we have to copy the host array
    // to a device-side buffer first — writing to fab(i,j,k) directly from host
    // would segfault (T4, A100, etc.) because the Array4<int> view points at
    // device memory.
#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceVector<int> dvec(total);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host_ptr, host_ptr + total, dvec.begin());
    amrex::Gpu::streamSynchronize();
    const int* src_ptr = dvec.data();
#else
    const int* src_ptr = host_ptr;
#endif

    const int nx_l = nx;
    const int ny_l = ny;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*(img->mf), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& fab = img->mf->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const std::size_t idx = static_cast<std::size_t>(k) * static_cast<std::size_t>(ny_l) *
                                        static_cast<std::size_t>(nx_l) +
                                    static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_l) +
                                    static_cast<std::size_t>(i);
            fab(i, j, k) = src_ptr[idx];
        });
    }
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();
#endif

    img->mf->FillBoundary(img->geom.periodicity());
    return img;
}

// =========================================================================
// PYBIND11_MODULE
// =========================================================================
PYBIND11_MODULE(_core, m) {
    m.doc() = "OpenImpala C++ backend — low-level bindings for transport property "
              "computation on 3-D voxel images of porous microstructures.";

    // --- AMReX lifecycle ---
    m.def("init_amrex", &init_amrex,
          "Initialise the AMReX runtime (no-op if already initialised).");
    m.def("finalize_amrex", &finalize_amrex,
          "Shut down the AMReX runtime (no-op if not initialised).");
    m.def("amrex_initialized", &amrex_initialized,
          "Return True if the AMReX runtime is currently active.");
    m.def("build_info", &build_info,
          "Return a dict of compile-time feature flags (cuda, openmp, mpi, tiny_profile, "
          "hypre_cuda) plus runtime GPU device count.  Use this to verify which wheel "
          "is installed (CPU vs. CUDA) — critical for Colab users who need the GPU wheel "
          "to actually use their T4/A100.");

    // --- VoxelImage opaque handle ---
    py::class_<OpenImpala::VoxelImage, std::shared_ptr<OpenImpala::VoxelImage>>(
        m, "VoxelImage",
        "Opaque container for a 3-D voxel image stored natively in AMReX memory.\n\n"
        "Create from a NumPy array via ``VoxelImage.from_numpy(arr)`` or receive\n"
        "one from ``read_image()``.  Pass to solver functions directly.")

        .def_static("from_numpy", &voxelimage_from_numpy, py::arg("arr"),
                    py::arg("max_grid_size") = 32,
                    "Construct a VoxelImage from a 3-D int32 NumPy array (Z, Y, X order).")

        .def("__repr__", [](const OpenImpala::VoxelImage& v) {
            if (!v.mf)
                return std::string("<VoxelImage (empty)>");
            const auto& bx = v.ba.minimalBox();
            return "<VoxelImage " + std::to_string(bx.length(0)) + "x" +
                   std::to_string(bx.length(1)) + "x" + std::to_string(bx.length(2)) + ">";
        });

    // Register enums first (used by everything else)
    init_enums(m);

    // I/O readers
    init_io(m);

    // Lightweight property calculators (VolumeFraction, PercolationCheck)
    init_props(m);

    // Heavy solvers (TortuosityHypre, TortuosityDirect, EffectiveDiffusivityHypre)
    init_solvers(m);

    // Configuration and output helpers (PhysicsConfig, ResultsJSON, CathodeWrite)
    init_config(m);
}
