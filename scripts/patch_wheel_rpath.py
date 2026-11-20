#!/usr/bin/env python3
"""Append `<env>/lib` (`$ORIGIN/../../..`) to `_core*.so` DT_RUNPATH inside repaired wheels.

`auditwheel repair` rewrites `_core*.so`'s RUNPATH to `$ORIGIN/../openimpala.libs`
(or `$ORIGIN/../openimpala_cuda.libs`).  Appending `$ORIGIN/../../..` and
`$ORIGIN/../../../openmpi` via `patchelf --set-rpath` (which writes ELF
`DT_RUNPATH` rather than `DT_RPATH`) lets `pip install openimpala openmpi` or
Conda environments resolve `libmpi.so.40` automatically without `LD_LIBRARY_PATH`,
while ensuring HPC `module load openmpi` (`LD_LIBRARY_PATH`) always takes precedence.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

EXTRA_RUNPATH = ("$ORIGIN/../../..", "$ORIGIN/../../../openmpi")


def _record_digest(data: bytes) -> str:
    b64 = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={b64}"


def patch_wheel(whl_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        unpack_dir = Path(tmp) / "unpacked"
        with zipfile.ZipFile(whl_path, "r") as zf:
            zf.extractall(unpack_dir)

        so_files = list(unpack_dir.glob("openimpala/_core*.so"))
        if not so_files:
            print(f"[skip] No _core*.so in {whl_path.name}")
            return

        patched_rel: dict[str, tuple[str, str]] = {}
        for so in so_files:
            cur = subprocess.check_output(["patchelf", "--print-rpath", str(so)], text=True).strip()
            parts = [p for p in cur.split(":") if p]
            for extra in EXTRA_RUNPATH:
                if extra not in parts:
                    parts.append(extra)
            new_rpath = ":".join(parts)
            subprocess.check_call(["patchelf", "--set-rpath", new_rpath, str(so)])
            verified = subprocess.check_output(["patchelf", "--print-rpath", str(so)], text=True).strip()
            rel = so.relative_to(unpack_dir).as_posix()
            raw = so.read_bytes()
            patched_rel[rel] = (_record_digest(raw), str(len(raw)))
            print(f"[patched] {whl_path.name}:{rel} RUNPATH -> {verified}")

        record_files = list(unpack_dir.glob("*.dist-info/RECORD"))
        if len(record_files) != 1:
            raise RuntimeError(f"Expected 1 RECORD file in {whl_path.name}, found {record_files}")
        rec_path = record_files[0]
        rows: list[list[str]] = []
        with rec_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                if row and row[0] in patched_rel:
                    digest, size_s = patched_rel[row[0]]
                    rows.append([row[0], digest, size_s])
                else:
                    rows.append(row)
        with rec_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerows(rows)

        tmp_whl = Path(tmp) / whl_path.name
        with zipfile.ZipFile(tmp_whl, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(p for p in unpack_dir.rglob("*") if p.is_file()):
                zf.write(file_path, file_path.relative_to(unpack_dir).as_posix())
        whl_path.write_bytes(tmp_whl.read_bytes())


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: patch_wheel_rpath.py <wheel_or_dir> [...]", file=sys.stderr)
        return 2
    for arg in argv:
        target = Path(arg)
        wheels = sorted(target.glob("*.whl")) if target.is_dir() else [target]
        for whl in wheels:
            patch_wheel(whl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
