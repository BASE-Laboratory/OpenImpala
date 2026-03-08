"""Tests for the CLI argument parser (does not invoke solvers)."""

from openimpala.cli import _build_parser


def test_parser_version():
    parser = _build_parser()
    # Just ensure it parses without error
    args = parser.parse_args(["vf", "dummy.tif", "--phase", "1"])
    assert args.command == "vf"
    assert args.phase == 1
    assert args.image == "dummy.tif"


def test_parser_analyze():
    parser = _build_parser()
    args = parser.parse_args([
        "analyze", "micro.tif",
        "--phase", "0",
        "--direction", "z",
        "--solver", "pcg",
    ])
    assert args.command == "analyze"
    assert args.direction == "z"
    assert args.solver == "pcg"


def test_parser_percolation():
    parser = _build_parser()
    args = parser.parse_args(["percolation", "data.h5", "--direction", "y"])
    assert args.command == "percolation"
    assert args.direction == "y"
