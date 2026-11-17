"""Tests for the CLI argument parser and command dispatch."""

import json
import pytest

import openimpala as oi
from openimpala.cli import _build_parser, _cmd_analyze
from openimpala.facade import TortuosityResult, VolumeFractionResult


def test_parser_version(capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == f"openimpala {oi.__version__}"


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


def test_cmd_analyze_mlmg_and_non_percolating(monkeypatch, capsys):
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
