"""Tests for v5.0 UX hardening: OpenMPI diagnostics, explicit phase API, and downsampling."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
from pathlib import Path
import shutil
import warnings
import zipfile

import numpy as np
import pytest

import openimpala as oi
from openimpala import session as sess_mod


class TestV500ImportFallbackAndRpath:
    """Verify loud fallback warnings when _core.so fails to load and wheel RUNPATH patching."""

    def test_broken_core_on_disk_warns_once_with_openmpi_remedy(self, monkeypatch):
        monkeypatch.delenv("OPENIMPALA_BACKEND", raising=False)
        monkeypatch.setattr(oi, "_core_binary_exists", lambda: True)
        monkeypatch.setattr(
            oi,
            "_probe_core",
            lambda: (
                False,
                "ImportError: libmpi.so.40: cannot open shared object file: No such file or directory",
            ),
        )
        with pytest.warns(RuntimeWarning, match=r"pip install openmpi"):
            oi._warn_if_compiled_core_broken()

    def test_pure_backend_or_pure_wheel_does_not_warn(self, monkeypatch):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            monkeypatch.setenv("OPENIMPALA_BACKEND", "pure")
            monkeypatch.setattr(oi, "_core_binary_exists", lambda: True)
            oi._warn_if_compiled_core_broken()

            monkeypatch.delenv("OPENIMPALA_BACKEND", raising=False)
            monkeypatch.setattr(oi, "_core_binary_exists", lambda: False)
            oi._warn_if_compiled_core_broken()

        assert len(caught) == 0

    def test_session_verbose_flag_forwards_to_init_amrex(self, monkeypatch):
        seen_verbose: list[bool] = []
        monkeypatch.setattr(sess_mod, "_requested_backend", lambda: "auto")
        monkeypatch.setattr(sess_mod, "_probe_core", lambda: (True, None))

        class _FakeCore:
            @staticmethod
            def init_amrex(verbose: bool = False) -> None:
                seen_verbose.append(bool(verbose))

            @staticmethod
            def amrex_initialized() -> bool:
                return False

            @staticmethod
            def finalize_amrex() -> None:
                pass

        import sys

        monkeypatch.setitem(sys.modules, "openimpala._core", _FakeCore)
        saved_depth = sess_mod.Session._depth
        saved_pure = sess_mod.Session._pure_python
        saved_verb = sess_mod.Session._verbose
        try:
            sess_mod.Session._depth = 0
            with oi.Session():
                pass
            with oi.Session(verbose=True):
                pass
            assert seen_verbose == [False, True]
        finally:
            sess_mod.Session._depth = saved_depth
            sess_mod.Session._pure_python = saved_pure
            sess_mod.Session._verbose = saved_verb

    def test_patch_wheel_rpath_updates_record_digest(self, tmp_path, monkeypatch):
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "patch_wheel_rpath.py"
        spec = importlib.util.spec_from_file_location("patch_wheel_rpath", script_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        whl_path = tmp_path / "openimpala-5.0.0-cp312-cp312-manylinux_2_28_x86_64.whl"
        so_rel = "openimpala/_core.cpython-312-x86_64-linux-gnu.so"
        orig_bytes = b"\x7fELF_FAKE_CORE_BINARY_BEFORE"
        orig_b64 = base64.urlsafe_b64encode(hashlib.sha256(orig_bytes).digest()).rstrip(b"=").decode()

        with zipfile.ZipFile(whl_path, "w") as zf:
            zf.writestr(so_rel, orig_bytes)
            zf.writestr("openimpala/__init__.py", "# init\n")
            zf.writestr(
                "openimpala-5.0.0.dist-info/RECORD",
                f"{so_rel},sha256={orig_b64},{len(orig_bytes)}\n"
                "openimpala/__init__.py,sha256=abc,7\n"
                "openimpala-5.0.0.dist-info/RECORD,,\n",
            )

        state = {"rpath": "$ORIGIN/../openimpala.libs"}

        def fake_check_output(cmd, text=True):
            assert cmd[:2] == ["patchelf", "--print-rpath"]
            return state["rpath"] + "\n"

        def fake_check_call(cmd):
            assert cmd[:2] == ["patchelf", "--set-rpath"]
            state["rpath"] = cmd[2]
            Path(cmd[3]).write_bytes(orig_bytes + b"_PATCHED")

        monkeypatch.setattr(mod.subprocess, "check_output", fake_check_output)
        monkeypatch.setattr(mod.subprocess, "check_call", fake_check_call)

        assert mod.main([str(whl_path)]) == 0
        assert "$ORIGIN/../../.." in state["rpath"]
        assert "$ORIGIN/../../../openmpi" in state["rpath"]

        with zipfile.ZipFile(whl_path, "r") as zf:
            patched_so = zf.read(so_rel)
            rec_text = zf.read("openimpala-5.0.0.dist-info/RECORD").decode("utf-8")

        exp_b64 = base64.urlsafe_b64encode(hashlib.sha256(patched_so).digest()).rstrip(b"=").decode()
        rows = {r[0]: r for r in csv.reader(rec_text.splitlines()) if r}
        assert rows[so_rel] == [so_rel, f"sha256={exp_b64}", str(len(patched_so))]


class TestV500RequiredPhaseAndDownsample:
    """Verify explicit phase requirement across all functions and downsample tie/truncation rules."""

    def test_all_phase_functions_require_explicit_phase(self):
        vol = np.ones((12, 12, 12), dtype=np.int32)

        with pytest.raises(TypeError):
            oi.volume_fraction(vol)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            oi.percolation_check(vol)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="phase"):
            oi.tortuosity(vol)
        with pytest.raises(TypeError, match="phase"):
            oi.tortuosity_all(vol)
        with pytest.raises(TypeError, match="phase"):
            oi.effective_diffusivity(vol)
        with pytest.raises(TypeError):
            oi.connected_components(vol)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            oi.particle_size_distribution(vol)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            oi.through_thickness_profile(vol)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="phase"):
            oi.rev_study(vol, sizes=[8])

    def test_multiphase_calls_do_not_require_dummy_phase(self):
        lam = np.ones((8, 8, 8), dtype=np.int32)
        lam[:, :2, :] = 0
        pmap = {0: 1.0, 1: 0.1}

        res_tau = oi.tortuosity(lam, direction="x", phase_diffusivities=pmap)
        assert res_tau.effective_diffusivity == pytest.approx(0.325, abs=1e-5)

        all_tau = oi.tortuosity_all(lam, directions=("x", "y"), phase_diffusivities=pmap)
        assert all_tau["x"] is not None and all_tau["y"] is not None
        assert all_tau["x"].effective_diffusivity == pytest.approx(0.325, abs=1e-5)
        assert all_tau["y"].effective_diffusivity == pytest.approx(1.0 / 7.75, abs=1e-4)

        if oi.build_info()["backend"] != "cpp-cuda":
            res_deff = oi.effective_diffusivity(lam, phase_diffusivities=pmap)
            assert res_deff.tensor[0, 0] == pytest.approx(0.325, abs=1e-5)
            assert res_deff.tensor[1, 1] == pytest.approx(1.0 / 7.75, abs=1e-4)

    def test_downsample_volume_truncation_warning_and_tie_break(self):
        odd = np.zeros((5, 6, 7), dtype=np.int32)
        with pytest.warns(UserWarning, match=r"\(5, 6, 7\).*\(2, 3, 3\)"):
            ds_odd = oi.downsample_volume(odd, factor=2)
        assert ds_odd.shape == (2, 3, 3)

        # Exact multiple -> zero warnings
        even = np.zeros((4, 4, 4), dtype=np.int32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds_even = oi.downsample_volume(even, factor=2)
        assert ds_even.shape == (2, 2, 2)
        assert len(caught) == 0

        # Exact 4-vs-4 tie in a single 2x2x2 block: lowest->0, highest->1
        tie = np.zeros((2, 2, 2), dtype=np.int32)
        tie[0, :, :] = 1
        assert int(oi.downsample_volume(tie, factor=2, tie_break="lowest")[0, 0, 0]) == 0
        assert int(oi.downsample_volume(tie, factor=2, tie_break="highest")[0, 0, 0]) == 1

        with pytest.raises(ValueError, match="tie_break"):
            oi.downsample_volume(tie, factor=2, tie_break="random")
