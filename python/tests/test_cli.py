"""Tests for the CLI argument parser and command dispatch."""

import json
import struct

import numpy as np
import pytest

import openimpala as oi
from openimpala.cli import _build_parser, _cmd_analyze, main
from openimpala.facade import PercolationResult, TortuosityResult, VolumeFractionResult


def _write_tiny_tiff(path, vol: np.ndarray) -> None:
    """Write a minimal uncompressed little-endian 3-D uint8 Classic TIFF."""
    arr = np.ascontiguousarray(vol, dtype="<u1")
    nz, ny, nx = arr.shape
    chunks = [b"II" + struct.pack("<HI", 42, 8)]
    pos = 8
    for z in range(nz):
        raw = arr[z].tobytes()
        pad = b"\x00" if (len(raw) % 2) else b""
        data_off = pos + 2 + 10 * 12 + 4
        next_ifd = (data_off + len(raw) + len(pad)) if z < nz - 1 else 0
        entries = [
            (256, 4, struct.pack("<I", nx)),
            (257, 4, struct.pack("<I", ny)),
            (258, 3, struct.pack("<HH", 8, 0)),
            (259, 3, struct.pack("<HH", 1, 0)),
            (262, 3, struct.pack("<HH", 1, 0)),
            (273, 4, struct.pack("<I", data_off)),
            (277, 3, struct.pack("<HH", 1, 0)),
            (278, 4, struct.pack("<I", ny)),
            (279, 4, struct.pack("<I", len(raw))),
            (339, 3, struct.pack("<HH", 1, 0)),
        ]
        ifd = struct.pack("<H", len(entries))
        for tag, ftype, val4 in entries:
            ifd += struct.pack("<HHI4s", tag, ftype, 1, val4)
        ifd += struct.pack("<I", next_ifd)
        chunks.extend([ifd, raw, pad])
        pos = next_ifd
    path.write_bytes(b"".join(chunks))


def test_parser_version(capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == f"openimpala {oi.__version__}"


def test_main_no_command_prints_help(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "usage: openimpala" in out


def test_parser_vf_defaults_and_grid_size():
    parser = _build_parser()
    args = parser.parse_args(["vf", "dummy.tif", "--phase", "1"])
    assert args.command == "vf"
    assert args.phase == 1
    assert args.image == "dummy.tif"
    assert args.max_grid_size == "auto"

    args_int = parser.parse_args(["vf", "dummy.tif", "--max-grid-size", "64"])
    assert args_int.max_grid_size == 64

    with pytest.raises(SystemExit):
        parser.parse_args(["vf", "dummy.tif", "--max-grid-size", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["vf", "dummy.tif", "--max-grid-size", "bad"])


def test_parser_analyze():
    parser = _build_parser()
    args_default = parser.parse_args(["analyze", "micro.tif"])
    assert args_default.solver == "auto"
    assert args_default.preconditioner == "smg"
    assert args_default.max_grid_size == "auto"

    args = parser.parse_args([
        "analyze", "micro.tif",
        "--phase", "0",
        "--direction", "z",
        "--solver", "mlmg",
        "--preconditioner", "pfmg",
    ])
    assert args.command == "analyze"
    assert args.direction == "z"
    assert args.solver == "mlmg"
    assert args.preconditioner == "pfmg"


def test_parser_percolation():
    parser = _build_parser()
    args = parser.parse_args(["percolation", "data.h5", "--direction", "y"])
    assert args.command == "percolation"
    assert args.direction == "y"
    assert args.max_grid_size == "auto"


def test_cmd_analyze_mlmg_and_error_paths(monkeypatch, capsys):
    parser = _build_parser()
    args = parser.parse_args(["analyze", "sample.tif", "--solver", "mlmg", "--direction", "z"])

    monkeypatch.setattr(oi, "read_image", lambda *a, **kw: (object(), "fake_img"))
    monkeypatch.setattr(
        oi,
        "volume_fraction",
        lambda img, phase=0: VolumeFractionResult(phase_count=50, total_count=100, fraction=0.5),
    )
    monkeypatch.setattr(
        oi,
        "tortuosity",
        lambda img, **kw: TortuosityResult(
            tortuosity=1.42,
            solver_converged=True,
            iterations=11,
            residual_norm=3.1e-9,
            flux_in=0.35,
            flux_out=0.35,
            active_volume_fraction=0.48,
        ),
    )

    rc = _cmd_analyze(args)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["percolates"] is True
    assert payload["tortuosity"] == pytest.approx(1.42)
    assert payload["iterations"] == 11
    assert payload["active_volume_fraction"] == pytest.approx(0.48)

    # Non-percolating path
    def _raise_perc(*a, **kw):
        raise oi.PercolationError("disconnected")

    monkeypatch.setattr(oi, "tortuosity", _raise_perc)
    rc2 = _cmd_analyze(args)
    assert rc2 == 0
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["percolates"] is False
    assert payload2["tortuosity"] is None

    # ConvergenceError path falls back to percolation_check and records error message
    def _raise_conv(*a, **kw):
        raise oi.ConvergenceError("stagnated at 1e-4")

    monkeypatch.setattr(oi, "tortuosity", _raise_conv)
    monkeypatch.setattr(
        oi,
        "percolation_check",
        lambda img, **kw: PercolationResult(
            percolates=True, active_volume_fraction=0.48, direction="Z"
        ),
    )
    rc3 = _cmd_analyze(args)
    assert rc3 == 0
    payload3 = json.loads(capsys.readouterr().out)
    assert payload3["percolates"] is True
    assert payload3["active_volume_fraction"] == pytest.approx(0.48)
    assert "stagnated" in payload3["error"]


def test_main_end_to_end_subcommands(tmp_path, capsys):
    vol = np.zeros((8, 8, 8), dtype=np.uint8)
    vol[:, :, :4] = 200  # Straight half-domain channel along Z and Y
    tif_path = tmp_path / "channel.tif"
    _write_tiny_tiff(tif_path, vol)

    # 1. openimpala vf
    assert main(["vf", str(tif_path), "--phase", "1", "--threshold", "100"]) == 0
    vf_json = json.loads(capsys.readouterr().out)
    assert vf_json["volume_fraction"] == pytest.approx(0.5)
    assert vf_json["phase_count"] == 256

    # 2. openimpala percolation
    assert main(["percolation", str(tif_path), "--phase", "1", "--threshold", "100", "--direction", "z"]) == 0
    perc_json = json.loads(capsys.readouterr().out)
    assert perc_json["percolates"] is True
    assert perc_json["direction"] == "Z"

    # 3. openimpala analyze with isotropic --voxel-size and --output file
    out_iso = tmp_path / "res_iso.json"
    assert main([
        "analyze", str(tif_path),
        "--phase", "1",
        "--threshold", "100",
        "--direction", "z",
        "--voxel-size", "0.5",
        "--output", str(out_iso),
    ]) == 0
    capsys.readouterr()
    iso_data = json.loads(out_iso.read_text())
    assert iso_data["percolates"] is True
    assert iso_data["voxel_size"] == pytest.approx([0.5, 0.5, 0.5])
    assert iso_data["tortuosity"] == pytest.approx(1.0, rel=1e-5)

    # 4. openimpala analyze with anisotropic 3-component --voxel-size DZ DY DX
    assert main([
        "analyze", str(tif_path),
        "--phase", "1",
        "--threshold", "100",
        "--direction", "z",
        "--voxel-size", "2.0", "1.5", "1.0",
    ]) == 0
    aniso_data = json.loads(capsys.readouterr().out)
    assert aniso_data["voxel_size"] == pytest.approx([2.0, 1.5, 1.0])
    assert aniso_data["tortuosity"] == pytest.approx(1.0, rel=1e-5)

    # 5. Invalid --voxel-size count (2 values) raises SystemExit
    with pytest.raises(SystemExit):
        main(["analyze", str(tif_path), "--voxel-size", "1.0", "2.0"])


def test_build_info_fallback_outside_session(monkeypatch):
    import openimpala.facade as facade_mod

    monkeypatch.setattr(facade_mod, "_is_pure_python", lambda: False)
    monkeypatch.setattr(facade_mod, "_has_core", lambda: False)
    info = oi.build_info()
    assert info["backend"] == "pure-python"
    assert info["version"] == oi.__version__


def test_analyze_downsample_deff_and_bpx_flags(tmp_path, capsys):
    vol = np.zeros((8, 8, 8), dtype=np.uint8)
    vol[:, :, :4] = 200  # Straight half-domain channel along Z and Y (blocked along X)
    tif_path = tmp_path / "channel_bpx.tif"
    _write_tiny_tiff(tif_path, vol)

    cli_args = [
        "analyze",
        str(tif_path),
        "--phase",
        "1",
        "--threshold",
        "100",
        "--direction",
        "z",
        "--voxel-size",
        "0.5",
        "--downsample",
        "2",
        "--bpx-electrode",
        "negative",
    ]
    with_deff = oi.build_info()["backend"] != "cpp-cuda"
    if with_deff:
        cli_args.append("--deff")

    assert main(cli_args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["shape"] == [4, 4, 4]
    assert payload["voxel_size"] == pytest.approx([1.0, 1.0, 1.0])
    assert payload["tortuosity"] == pytest.approx(1.0, rel=1e-5)
    assert payload["macmullin_number"] == pytest.approx(2.0, rel=1e-5)
    assert payload["bruggeman_exponent"] == pytest.approx(1.0, rel=1e-5)
    assert payload["bpx"]["electrode"] == "negative"
    neg = payload["bpx"]["Parameterisation"]["Negative electrode"]
    assert neg["Porosity"] == pytest.approx(0.5)
    assert neg["Transport efficiency"] == pytest.approx(0.5, rel=1e-5)
    assert neg["Bruggeman coefficient (electrolyte)"] == pytest.approx(1.0, rel=1e-5)
    if with_deff:
        assert payload["deff_diagonal"] == pytest.approx([0.0, 0.5, 0.5], abs=1e-5)

    with pytest.raises(SystemExit):
        main(["analyze", str(tif_path), "--downsample", "0"])



